use braine::{storage, substrate::Brain};
use std::io::{self, Read, Write};

pub const MAGIC: &[u8; 8] = b"BRSTATE1";
pub const VERSION_V1: u32 = 1;

/// Tags for chunks in the persisted daemon state file.
const TAG_BRAIN_IMAGE: [u8; 4] = *b"BIMG";
const TAG_EXPERTS_STATE: [u8; 4] = *b"EXPT";

pub struct LoadedState {
    pub brain: Brain,
    pub experts_state: Option<Vec<u8>>,
}

pub fn save_state_to<W: Write>(w: &mut W, brain: &Brain, experts_state: &[u8]) -> io::Result<()> {
    w.write_all(MAGIC)?;
    storage::write_u32_le(w, VERSION_V1)?;

    let mut brain_bytes: Vec<u8> = Vec::new();
    brain.save_image_to(&mut brain_bytes)?;
    storage::write_chunk(w, TAG_BRAIN_IMAGE, &brain_bytes)?;

    storage::write_chunk(w, TAG_EXPERTS_STATE, experts_state)?;

    Ok(())
}

pub fn load_state_from<R: Read>(r: &mut R) -> io::Result<LoadedState> {
    let magic = storage::read_exact::<8, _>(r)?;
    if &magic != MAGIC {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "bad state magic",
        ));
    }

    let version = storage::read_u32_le(r)?;
    if version != VERSION_V1 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "unsupported state version",
        ));
    }

    let mut brain_img: Option<Vec<u8>> = None;
    let mut experts_state: Option<Vec<u8>> = None;

    loop {
        let (tag, len) = match storage::read_chunk_header(r) {
            Ok(v) => v,
            Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => break,
            Err(e) => return Err(e),
        };

        let mut take = r.take(len as u64);
        let mut buf = vec![0u8; len as usize];
        take.read_exact(&mut buf)?;

        if tag == TAG_BRAIN_IMAGE {
            brain_img = Some(buf);
        } else if tag == TAG_EXPERTS_STATE {
            experts_state = Some(buf);
        }
    }

    let brain_img = brain_img
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "missing brain image chunk"))?;
    let mut cursor = std::io::Cursor::new(brain_img);
    let brain = Brain::load_image_from(&mut cursor)?;

    Ok(LoadedState {
        brain,
        experts_state,
    })
}
