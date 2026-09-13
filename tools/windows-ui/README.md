# Windows rendering verification

`windows-ui.yml` runs installed Microsoft Edge on a standard Windows 11 ARM
runner and installed Google Chrome on a standard Windows Server 2025 x64 runner.
It builds the same reviewed, derived-only package as Pages, then serves that
package on the Windows machine. No user-agent override or browser download is
used. Standard hosted runners are free for this public repository; the workflow
refuses private repositories. Artifacts expire after one day.

The report records the actual OS, architecture, browser version, release manifest
hash, loaded font faces, and CDP-reported fonts used for Korean glyphs. It checks
card padding and text bounds, page overflow at 1440/1280/390/320 CSS pixels, dark
mode, the allocation summary width, navigation, and controls changing results.
PNG screenshots come directly from those Windows browser sessions. Screenshot
review remains a separate step from automated checks.

Run in PowerShell on Windows after preparing `dist/public-dashboard`:

```powershell
npm ci --prefix tools/windows-ui --ignore-scripts --no-audit --no-fund
node scripts/verify_windows_ui.cjs --browser msedge --package dist/public-dashboard --output build/windows-ui/msedge
```

Use `--browser chrome` for Chrome. `--base-url https://sonchanggi.github.io/regime/`
checks the deployed page and first requires its publication manifest to match
the local package. The workflow dispatch `site_url` input provides this option.

For development only, `--allow-non-windows` permits a local rehearsal, explicitly
marked `actualWindows: false` in its report. CI never uses this option, and such
a run does not satisfy Windows verification.
