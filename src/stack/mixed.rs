//! Consecutive runs of INT8 and BF16 blocks, preserving global conditioning,
//! adapter indices and graph/cache ranges across precision boundaries.
use super::*;

pub(crate) struct MixedStack {
    runs: Vec<Stack>,
    layers: usize,
    calls: usize,
}

impl MixedStack {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        c: &Compiler,
        stream: &mut hrx::Stream,
        d: StackDims,
        tokens: usize,
        layers: usize,
        w: &Weights,
        prefix: impl Fn(usize) -> String,
        qk_weights: bool,
        ones_head: Arc<hrx::Buffer>,
        tag: &str,
    ) -> Result<Self> {
        let quant = |i| w.has(&format!("{}qkv.s", prefix(i)));
        let mut runs = Vec::new();
        let mut first = 0;
        while first < layers {
            let is_int8 = quant(first);
            let last = (first + 1..layers)
                .find(|&i| quant(i) != is_int8)
                .unwrap_or(layers);
            let mut dims = d.clone();
            dims.wbits = if is_int8 { 8 } else { 16 };
            dims.bf16 = !is_int8;
            let mut stack = Stack::new(
                c,
                stream,
                dims,
                tokens,
                last - first,
                w,
                |i| prefix(first + i),
                qk_weights,
                ones_head.clone(),
                tag,
            )?;
            stack.layer_offset = first;
            runs.push(stack);
            first = last;
        }
        Ok(Self {
            runs,
            layers,
            calls: 0,
        })
    }

    pub(crate) fn capacity(&self) -> usize {
        self.runs.iter().map(Stack::capacity).max().unwrap_or(0)
    }

    pub(crate) fn attach_adapter(
        &mut self,
        c: &Compiler,
        stream: &mut hrx::Stream,
        adapter: &crate::adapter::Adapter,
        refiner: bool,
    ) -> Result<()> {
        for stack in &mut self.runs {
            stack.attach_adapter(c, stream, adapter, refiner)?;
        }
        Ok(())
    }

    pub(crate) fn has_world_routing(&self) -> bool {
        self.runs.iter().any(Stack::has_world_routing)
    }

    pub(crate) fn world_routing(
        &mut self,
        c: &Compiler,
        stream: &mut hrx::Stream,
        codes: Option<&[i32]>,
    ) -> Result<()> {
        for stack in &mut self.runs {
            stack.world_routing(c, stream, codes)?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn forward<'a>(
        &mut self,
        stream: &mut hrx::Stream,
        prof: &mut Profile,
        x: View<'_>,
        cls: ClassRows<'_>,
        cos: View<'_>,
        sin: View<'_>,
        cond: &dyn Fn(usize) -> LayerCond<'a>,
        first: usize,
        last: Option<usize>,
    ) -> Result<()> {
        let dumping = env_once("H3_DUMP_BLOCKS").is_some() && {
            let call = env_once("H3_DUMP_CALL")
                .and_then(|v| v.parse().ok())
                .unwrap_or(0);
            let this = self.calls;
            self.calls += 1;
            this == call && first == 0
        };
        self.emit(
            &mut Sink::Stream(stream),
            prof,
            x,
            cls,
            cos,
            sin,
            cond,
            first,
            last,
            dumping,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn forward_cached<'a>(
        &mut self,
        stream: &mut hrx::Stream,
        prof: &mut Profile,
        recorded: &mut Option<hrx::GraphExec>,
        x: View<'a>,
        cls: ClassRows<'a>,
        cos: View<'a>,
        sin: View<'a>,
        cond: &dyn Fn(usize) -> LayerCond<'a>,
        first: usize,
        last: Option<usize>,
    ) -> Result<()> {
        if !crate::dispatch::graph_enabled(prof) {
            return self.forward(stream, prof, x, cls, cos, sin, cond, first, last);
        }
        if recorded.is_none() {
            let mut graph = stream.graph()?;
            self.emit(
                &mut Sink::Graph {
                    graph: &mut graph,
                    after: Default::default(),
                },
                prof,
                x,
                cls,
                cos,
                sin,
                cond,
                first,
                last,
                false,
            )?;
            *recorded = Some(graph.finish()?);
        }
        stream.launch(recorded.as_mut().expect("recorded above"))?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn emit<'a, 'g>(
        &'g self,
        sink: &mut Sink<'_, 'g>,
        prof: &mut Profile,
        x: View<'g>,
        cls: ClassRows<'g>,
        cos: View<'g>,
        sin: View<'g>,
        cond: &dyn Fn(usize) -> LayerCond<'a>,
        first: usize,
        last: Option<usize>,
        dumping: bool,
    ) -> Result<()>
    where
        'a: 'g,
    {
        let last = last.unwrap_or(self.layers);
        if first > last || last > self.layers {
            return err("block range exceeds mixed stack");
        }
        for stack in &self.runs {
            if let Some(range) = intersection(
                first..last,
                stack.layer_offset..stack.layer_offset + stack.layers,
            ) {
                let local_cond = |i| cond(stack.layer_offset + i);
                stack.emit(
                    sink,
                    prof,
                    x,
                    cls,
                    cos,
                    sin,
                    &local_cond,
                    range.start,
                    Some(range.end),
                    dumping,
                )?;
            }
        }
        Ok(())
    }
}

fn intersection(
    request: std::ops::Range<usize>,
    run: std::ops::Range<usize>,
) -> Option<std::ops::Range<usize>> {
    let start = request.start.max(run.start);
    let end = request.end.min(run.end);
    (start < end).then(|| start - run.start..end - run.start)
}

#[cfg(test)]
mod tests {
    use super::intersection;

    #[test]
    fn cache_ranges_preserve_layers_across_precision_boundaries() {
        let runs = [0..2, 2..48, 48..50];
        for request in [0..50, 0..1, 1..50, 2..48, 0..0, 49..50] {
            let visited: Vec<_> = runs
                .iter()
                .flat_map(|run| {
                    intersection(request.clone(), run.clone())
                        .into_iter()
                        .flatten()
                        .map(|i| i + run.start)
                })
                .collect();
            assert_eq!(visited, request.collect::<Vec<_>>());
        }
    }
}
