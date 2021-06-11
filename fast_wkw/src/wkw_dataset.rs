use lru::LruCache;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use wkwrap::{Dataset, Header, Result, Vec3};

use crate::wkw_file::WkwFile;

#[derive(Clone, Debug)]
pub struct WkwFileCache {
  inner: Arc<Mutex<LruCache<PathBuf, Arc<WkwFile>>>>,
}

impl WkwFileCache {
  pub fn new(cap: usize) -> Self {
    Self {
      inner: Arc::new(Mutex::new(LruCache::new(cap))),
    }
  }

  pub fn get_file(&self, path: &Path) -> Result<Arc<WkwFile>> {
    let mut cache = self.inner.lock().unwrap();
    let cached_file = cache.get(&PathBuf::from(path)).cloned();
    if let Some(cached_file) = cached_file {
      return Ok(cached_file);
    }
    let file = Arc::new(WkwFile::open(&path)?);
    cache.put(PathBuf::from(path), file.clone());
    Ok(file)
  }

  pub fn clear_prefix(&self, path_prefix: &Path) {
    let mut cache = self.inner.lock().unwrap();
    let paths_to_delete = cache
      .iter()
      .filter_map(|(path, _)| {
        if path.starts_with(path_prefix) {
          Some(path)
        } else {
          None
        }
      })
      .cloned()
      .collect::<Vec<_>>();
    for path in paths_to_delete.into_iter() {
      cache.pop(&path);
    }
  }
}

#[derive(Clone, Debug)]
pub struct WkwDataset {
  root: PathBuf,
  header: Header,
  file_cache: WkwFileCache,
}

impl WkwDataset {
  pub fn new(root: &Path, file_cache: WkwFileCache) -> Result<Self> {
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

    // try to open file, it is fine to return an empty bucket if the file is not present
    if let Ok(file) = self.file_cache.get_file(&file_path) {
      file.read_block(file_src_pos, buf)?;
    }
    Ok(())
  }
}
