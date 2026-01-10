# Deploying braine_web

This document explains how to build and deploy the fully in-browser `braine_web` application.

## Overview

`braine_web` is a Leptos CSR (Client-Side Rendering) application that runs entirely in the browser. It includes:
- An in-process `Brain` (no remote daemon)
- Shared game logic (`crates/shared/braine_games`)
- IndexedDB persistence for brain images
- Canvas-based visualizations for Pong and SpotXY

**GPU status**: The core `braine` substrate currently uses the `Scalar` execution tier in WASM builds. The `gpu` feature (based on `wgpu`) is designed for native targets and requires `std` + blocking APIs (`pollster::block_on`), which are not WASM-compatible. A future WebGPU port would require:
- Async `wgpu` initialization using `wasm-bindgen-futures`
- Adapting the GPU compute pipeline for WASM32 + WebGPU backend
- Feature-gating to avoid conflicts with native GPU builds

The web UI includes a runtime check for `navigator.gpu` availability and displays whether WebGPU is detected (currently informational only).

---

## Prerequisites

1. **Rust toolchain** with the `wasm32-unknown-unknown` target:
   ```bash
   rustup target add wasm32-unknown-unknown
   ```

2. **Trunk** (a WASM web application bundler):
   ```bash
   cargo install trunk
   ```

3. (Optional) **wasm-opt** from the [Binaryen](https://github.com/WebAssembly/binaryen) toolkit for size optimization:
   - On Ubuntu/Debian: `sudo apt install binaryen`
   - On macOS: `brew install binaryen`
   - On Windows: Download from the Binaryen releases page

---

## Building for Production

### 1. Navigate to the web crate
```bash
cd crates/braine_web
```

### 2. Build with Trunk
```bash
trunk build --release --features web
```

This produces optimized WASM + HTML + JS in `crates/braine_web/dist/`.

### 3. (Optional) Further optimize WASM binary size
After the Trunk build, run `wasm-opt` manually:
```bash
wasm-opt -Oz -o dist/braine_web_bg.wasm.opt dist/braine_web_bg.wasm
mv dist/braine_web_bg.wasm.opt dist/braine_web_bg.wasm
```

The `-Oz` flag applies aggressive size optimizations. Expect the final WASM size to be ~500KB–1.5MB (depending on optimization level and enabled features).

---

## Deployment Options

### GitHub Pages

1. **Enable GitHub Pages** in your repository settings:
   - Go to **Settings → Pages**
   - Source: **Deploy from a branch**
   - Branch: **`gh-pages`** (or a dedicated deployment branch)
   - Folder: `/ (root)`

2. **Copy build artifacts** to a deployment branch:
   ```bash
   # From the repo root:
   cd crates/braine_web
   trunk build --release --features web

   # Create/update deployment branch (assumes 'gh-pages' branch exists)
   git checkout gh-pages
   git pull origin gh-pages
   rm -rf *  # Clear old artifacts
   cp -r dist/* .
   git add .
   git commit -m "Deploy braine_web"
   git push origin gh-pages
   ```

3. **Access your site**:
   - `https://<your-username>.github.io/<repo-name>/`
   - (If deploying at the repo root, the URL is `https://<your-username>.github.io/<repo-name>/`)

**Note**: GitHub Pages requires the `index.html` to be at the root of the deployment folder. The Trunk build places everything in `dist/`, so copy the contents of `dist/` (not `dist/` itself) to the deployment branch root.

### Netlify / Vercel / Cloudflare Pages

These platforms support static site hosting and can build the WASM app directly from your repo.

**Netlify example**:
1. Connect your GitHub repo.
2. Set build command:
   ```bash
   cd crates/braine_web && trunk build --release --features web
   ```
3. Set publish directory: `crates/braine_web/dist`
4. Deploy.

**Vercel / Cloudflare Pages**: Similar configuration; specify the build command and output directory.

### Self-Hosted Static Server

Serve the `dist/` folder with any static file server (e.g., `nginx`, `caddy`, `python -m http.server`).

Example with Python:
```bash
cd crates/braine_web/dist
python3 -m http.server 8080
# Visit http://localhost:8080
```

**CORS / Security Headers**: Since `braine_web` uses IndexedDB and WASM, ensure your hosting environment:
- Serves with `Content-Type: application/wasm` for `.wasm` files
- Does not block `Cross-Origin-Opener-Policy` if needed by your hosting setup

---

## Optimization Tips

1. **Reduce WASM binary size**:
   - Use `--release` mode (includes LTO and opt-level 3).
   - Run `wasm-opt -Oz` as shown above.
   - Strip debug symbols: `wasm-strip dist/braine_web_bg.wasm` (requires `wabt` toolkit).

2. **Enable compression**:
   - Serve `.wasm` files with gzip or brotli compression (most static hosts support this automatically).

3. **Lazy loading**:
   - For large apps, consider splitting WASM modules or lazy-loading assets (Leptos + Trunk support code-splitting).

4. **IndexedDB caching**:
   - The app already uses IndexedDB for brain persistence. Consider adding a service worker for offline support and caching static assets.

---

## Browser Compatibility

- **Supported**: Modern browsers with WASM + IndexedDB support (Chrome 80+, Firefox 79+, Safari 14+, Edge 80+).
- **WebGPU**: Optional; detected at runtime but not yet integrated with the brain substrate.

---

## Troubleshooting

**Build fails with "features not enabled"**:
- Ensure you're building with `--features web`: `trunk build --release --features web`

**WASM load error in browser**:
- Check browser console for MIME type errors. Ensure the server serves `.wasm` files with `Content-Type: application/wasm`.

**IndexedDB not working**:
- Ensure HTTPS (GitHub Pages, Netlify, etc. provide HTTPS by default). Some browsers restrict IndexedDB over HTTP.

---

## Future Work

- **WebGPU integration**: Port the `gpu` feature to async WASM-compatible `wgpu` for hardware-accelerated substrate updates.
- **Service Worker**: Add offline support and cache static assets for faster load times.
- **Advanced visualizations**: Integrate WebGL/WebGPU-based real-time brain state rendering (spike rasters, phase plots).

---

## Example Deployment Script

```bash
#!/usr/bin/env bash
# deploy-web.sh
# Builds and deploys braine_web to GitHub Pages (gh-pages branch).

set -e

cd "$(dirname "$0")/../crates/braine_web" || exit 1

echo "Building braine_web (release + wasm-opt)..."
trunk build --release --features web

if command -v wasm-opt &> /dev/null; then
  echo "Optimizing WASM binary with wasm-opt..."
  wasm-opt -Oz -o dist/braine_web_bg.wasm.opt dist/braine_web_bg.wasm
  mv dist/braine_web_bg.wasm.opt dist/braine_web_bg.wasm
else
  echo "wasm-opt not found; skipping additional optimization."
fi

echo "Deploying to gh-pages branch..."
git checkout gh-pages
git pull origin gh-pages
rm -rf *
cp -r dist/* .
git add .
git commit -m "Deploy braine_web $(date +'%Y-%m-%d %H:%M')"
git push origin gh-pages
git checkout main

echo "Deployment complete. Visit your GitHub Pages URL."
```

Save as `scripts/deploy-web.sh`, make executable (`chmod +x scripts/deploy-web.sh`), and run from the repo root.

---

## Summary

- **Build**: `trunk build --release --features web` in `crates/braine_web/`
- **Deploy to GitHub Pages**: Copy `dist/*` to `gh-pages` branch root
- **Optimize**: Use `wasm-opt -Oz` for smaller binaries
- **GPU**: Currently CPU-only (`Scalar` tier); WebGPU is a future enhancement
