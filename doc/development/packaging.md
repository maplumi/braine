# Packaging for Windows

## Portable ZIP (no installer)
1. Build Windows binaries from Linux (cross):
   ```bash
   cargo build --release --target x86_64-pc-windows-gnu
   ```
2. Stage the files:
   ```bash
   mkdir -p dist/windows
   cp target/x86_64-pc-windows-gnu/release/braine_viz.exe dist/windows/
   cp target/x86_64-pc-windows-gnu/release/brained.exe dist/windows/
   cp target/x86_64-pc-windows-gnu/release/braine.exe dist/windows/
   cp braine_vis/assets/braine.ico dist/windows/
   printf "@echo off\nstart \"brained\" brained.exe\ntimeout /t 1 >NUL\nstart \"braine_viz\" braine_viz.exe\n" > dist/windows/run_braine.bat
   ```
3. Zip the payload:
   ```bash
   cd dist/windows
   7z a -tzip ../braine-portable.zip braine_viz.exe brained.exe braine.exe braine.ico run_braine.bat
   ```
4. On Windows, unzip and run `run_braine.bat` (make a shortcut to it if desired).

## Inno Setup installer (on Windows)
1. Copy the same payload from `dist/windows` to a Windows machine.
2. Open `dist/windows/braine.iss` in Inno Setup.
3. Compile to produce `braine-setup.exe` (adds Start Menu / optional desktop shortcut; launches daemon and UI post-install).

## Notes
- `braine_viz` is the UI, `brained` is the daemon, `braine` is the CLI/demo binary.
- For a one-click portable start, `run_braine.bat` launches `brained.exe`, waits 1s, then launches `braine_viz.exe`.
- If 7-Zip CLI is missing, install `p7zip-full` on Linux or the standard 7-Zip on Windows.
