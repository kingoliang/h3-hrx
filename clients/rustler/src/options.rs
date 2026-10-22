use rustler::{Atom, Decoder, NifResult, Term};

pub struct Options<'a> {
    term: Term<'a>,
    read: usize,
}
impl<'a> Options<'a> {
    pub fn new(term: Term<'a>) -> NifResult<Self> {
        term.map_size()?;
        Ok(Self { term, read: 0 })
    }
    pub fn get<T: Decoder<'a>>(&mut self, key: &str) -> NifResult<Option<T>> {
        let key = Atom::from_str(self.term.get_env(), key)?;
        match self.term.map_get(key) {
            Ok(value) => {
                self.read += 1;
                value.decode()
            }
            Err(_) => Ok(None),
        }
    }
    pub fn finish(self) -> NifResult<()> {
        if self.read == self.term.map_size()? {
            Ok(())
        } else {
            Err(rustler::Error::BadArg)
        }
    }
}

#[derive(Default)]
pub struct GenerateOptions {
    pub sampling: h3_hrx::SamplingOptions,
    pub bypass_upscale: bool,
}
impl GenerateOptions {
    pub fn decode(term: Term<'_>) -> NifResult<Self> {
        use h3_hrx::{Sampler, SamplingPreset, Scheduler};
        let mut o = Options::new(term)?;
        let preset = match o.get::<String>("preset")?.as_deref() {
            None => None,
            Some("fasth3") => Some(SamplingPreset::FastH3),
            _ => return Err(rustler::Error::BadArg),
        };
        let sampler = match o.get::<String>("sampler")?.as_deref() {
            None => None,
            Some("euler") => Some(Sampler::Euler),
            Some("res_multistep") => Some(Sampler::ResMultistep),
            _ => return Err(rustler::Error::BadArg),
        };
        let scheduler = match o.get::<String>("scheduler")?.as_deref() {
            None => None,
            Some("simple") => Some(Scheduler::Simple),
            Some("uniform") => Some(Scheduler::Uniform),
            _ => return Err(rustler::Error::BadArg),
        };
        let result = Self {
            sampling: h3_hrx::SamplingOptions {
                preset,
                sampler,
                scheduler,
                evaluations: o.get("evaluations")?,
                steps: o.get("steps")?,
                video_shift: o.get("video_shift")?,
                audio_shift: o.get("audio_shift")?,
            },
            bypass_upscale: o.get("bypass_upscale")?.unwrap_or(false),
        };
        o.finish()?;
        Ok(result)
    }
}
