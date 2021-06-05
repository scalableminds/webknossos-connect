use lru::LruCache;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, RwLock};
use wkwrap::{Dataset, Header, Result, Vec3};

use crate::wkw_file::File;

#[derive(Clone, Debug)]
pub struct FileCache {
  inner: Arc<Mutex<LruCache<PathBuf, Arc<RwLock<File>>>>>,
}

impl FileCache {
  pub fn new(cap: usize) -> Self {
    Self {
      inner: Arc::new(Mutex::new(LruCache::new(cap))),
    }
  }

  pub fn get_file(&self, path: &Path) -> Option<Arc<RwLock<File>>> {
    let mut cache = self.inner.lock().unwrap();
    let cached_file = cache.get(&PathBuf::from(path)).cloned();
    if let Some(cached_file) = cached_file {
      return Some(cached_file);
    }
    match File::open(&path) {
      Ok(file) => {
        let file = Arc::new(RwLock::new(file));
        cache.put(PathBuf::from(path), file.clone());
        Some(file)
      }
      Err(_) => None,
    }
  }
}

#[derive(Clone, Debug)]
pub struct CachedDataset {
  root: PathBuf,
  header: Header,
  file_cache: FileCache,
}

impl CachedDataset {
  pub fn new(root: &Path, file_cache: FileCache) -> Result<Self> {
    if !root.is_dir() {
      return Err(format!("Dataset root {:?} is not a directory", &root));
    }

    // read required header file
    let header = Dataset::new(root)?.header().clone();

    Ok(Self {
      root: root.to_owned(),
      header,
      file_cache,
    })
  }

  pub fn header(&self) -> &Header {
    &self.header
  }

  pub fn read_block(&self, src_pos: Vec3, buf: &mut [u8]) -> Result<()> {
    let file_len_vx_log2 = self.header.file_len_vx_log2() as u32;
    let file_ids = src_pos >> file_len_vx_log2;

    assert_eq!(src_pos.x % 32, 0);
    assert_eq!(src_pos.y % 32, 0);
    assert_eq!(src_pos.z % 32, 0);

    let file_path = self
      .root
      .join(format!("z{}", file_ids.z))
      .join(format!("y{}", file_ids.y))
      .join(format!("x{}.wkw", file_ids.x));

    let file_src_pos = src_pos - (file_ids << file_len_vx_log2);

    // try to open file
    if let Some(file) = self.file_cache.get_file(&file_path) {
      file.read().unwrap().read_block(file_src_pos, buf)?;
    }
    Ok(())
  }
}
