//! Cross-platform application paths

use std::fs;
use std::path::PathBuf;

#[derive(Debug, Clone)]
pub struct AppPaths {
    data_dir: PathBuf,
}

impl AppPaths {
    pub fn new() -> Result<Self, String> {
        let data_dir = Self::get_data_dir()?;

        // Ensure directory exists
        fs::create_dir_all(&data_dir)
            .map_err(|e| format!("Failed to create data directory: {}", e))?;

        Ok(Self { data_dir })
    }

    fn get_data_dir() -> Result<PathBuf, String> {
        let base = dirs::data_dir().ok_or("Could not determine data directory")?;
        Ok(base.join("braine"))
    }

    #[allow(dead_code)]
    pub fn data_dir(&self) -> &PathBuf {
        &self.data_dir
    }

    pub fn brain_file(&self) -> PathBuf {
        self.data_dir.join("braine.bbi")
    }

    pub fn runtime_state_file(&self) -> PathBuf {
        self.data_dir.join("runtime.json")
    }

    #[allow(dead_code)]
    pub fn config_file(&self) -> PathBuf {
        self.data_dir.join("config.json")
    }

    #[allow(dead_code)]
    pub fn log_file(&self) -> PathBuf {
        self.data_dir.join("brained.log")
    }
}
