<#
Runs the correctness tests and benchmarks with the `asm` feature on and off on the repository this
script sits in, times the run-time CPUID selection of the MULX/ADX kernel against a build with BMI2
and ADX at compile time, and with `-Zakura` also Zakura #701's GLV MSM with AVX2 on and off. Writes
raw logs, summary.txt and x86-bench-results.zip under .\x86-bench.

From the repository root:
    powershell -ExecutionPolicy Bypass -File scripts\x86_bench.ps1

`-Affinity` pins the timing runs to a hexadecimal CPU mask; `-Affinity 555` takes one logical CPU
on each of an i9-12900H's six P-cores (logical CPUs 0-11). Without it the runs are not pinned.
#>
param(
    [string]$Branch = "platform-specific",
    [string]$ZakuraCommit = "05dffadf7fd21f1d6adda7fd7ddc75159b70b2c0",
    [int]$Rounds = 2,
    [string]$Affinity = "",
    [string]$Threads = "1,4,6",
    [string]$MsmSizes = "256,1024,4096,16384",
    [switch]$SkipTests,
    [switch]$Zakura
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
[Console]::OutputEncoding = [Text.Encoding]::UTF8
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

$Root = Join-Path (Get-Location) "x86-bench"
$Out = Join-Path $Root "results"
New-Item -ItemType Directory -Force -Path $Out | Out-Null
$Summary = Join-Path $Out "summary.txt"
Set-Content -Path $Summary -Value "" -Encoding UTF8

function Say([string]$s) {
    Write-Host $s -ForegroundColor Cyan
    Add-Content -Path $Summary -Value $s -Encoding UTF8
}

function Get-Source([string]$Url, [string]$Name) {
    $dest = Join-Path $Root $Name
    if (Test-Path $dest) { return $dest }
    Say "Downloading $Url"
    $zip = "$dest.zip"
    $tmp = "$dest.tmp"
    Invoke-WebRequest -UseBasicParsing -Uri $Url -OutFile $zip
    New-Item -ItemType Directory -Force -Path $tmp | Out-Null
    # Windows' tar.exe; symbolic links in the archive fail without Developer Mode and are not needed.
    & cmd /c "tar -xf `"$zip`" -C `"$tmp`" 2>nul"
    Move-Item (Get-ChildItem $tmp -Directory | Select-Object -First 1).FullName $dest
    Remove-Item $tmp, $zip -Recurse -Force
    return $dest
}

# Runs `$Cmd` in `$Dir` through cmd.exe, echoes it, saves it to results\$Log and returns the exit code.
function Run([string]$Dir, [string]$Cmd, [string]$Log) {
    Write-Host "> $Cmd" -ForegroundColor DarkGray
    Push-Location $Dir
    try {
        $lines = @(& cmd /c "$Cmd 2>&1" | ForEach-Object { Write-Host $_; "$_" })
        $code = $LASTEXITCODE
    } finally {
        Pop-Location
    }
    Set-Content -Path (Join-Path $Out $Log) -Value $lines -Encoding UTF8
    return $code
}

$AllCpus = [System.Diagnostics.Process]::GetCurrentProcess().ProcessorAffinity
function Pin([bool]$On) {
    if ($Affinity -eq "") { return }
    $p = [System.Diagnostics.Process]::GetCurrentProcess()
    if ($On) { $p.ProcessorAffinity = [IntPtr]([Convert]::ToInt64($Affinity, 16)) } else { $p.ProcessorAffinity = $AllCpus }
}

function Median([double[]]$v) {
    $s = @($v | Sort-Object)
    if ($s.Count % 2) { return $s[[int][math]::Floor($s.Count / 2)] }
    return ($s[$s.Count / 2 - 1] + $s[$s.Count / 2]) / 2
}

function Fmt([double]$v) { return $v.ToString("0.###", [Globalization.CultureInfo]::InvariantCulture) }

function ToNs([double]$v, [string]$unit) {
    if ($unit -eq "" -or $unit -eq "x") { return $v }
    if ($unit -eq "ns") { return $v }
    if ($unit -eq "ms") { return $v * 1e6 }
    if ($unit -eq "s") { return $v * 1e9 }
    return $v * 1e3
}

# A number not inside an identifier, with an optional duration unit or an `x` ratio suffix.
$TokenRe = [regex]'(?<![A-Za-z_.\d])(\d+(?:\.\d+)?)(ns|\u00B5s|us|ms|s|x)?(?![A-Za-z_\d])'
$NoiseRe = [regex]'^\s*(test |warning|error|note:|help:|-->|\||=|\d+\s+\|)'

# The benchmark's own rows: lines between libtest's `running` and `test result` lines that hold a number.
function DataLines([string]$Log) {
    $path = Join-Path $Out $Log
    if (-not (Test-Path $path)) { return , @() }
    $rows = @()
    $inside = $false
    foreach ($l in Get-Content $path) {
        if ($l -match '^running \d+ tests?') { $inside = $true; continue }
        if ($l -match '^test result:') { $inside = $false; continue }
        if ($inside -and $l.Trim() -ne "" -and -not $NoiseRe.IsMatch($l) -and $TokenRe.IsMatch($l)) { $rows += $l }
    }
    return , $rows
}

# Pairs the rows of build A's and build B's logs by position. Numbers equal in every run stay as
# they are; the others become `[A a | B b | b/a x]`, medians over rounds, durations in ns.
function Compare-Builds([string]$Bench, [string]$A = "asm", [string]$B = "portable") {
    Say ""
    Say "== $Bench   [$A | $B | $B/$A]"
    $a = @(); $p = @()
    for ($r = 1; $r -le $Rounds; $r++) {
        $a += , (DataLines "$Bench.$A.$r.txt")
        $p += , (DataLines "$Bench.$B.$r.txt")
    }
    $all = $a + $p
    $n = $all[0].Count
    if ($n -eq 0 -or @($all | Where-Object { $_.Count -ne $n }).Count -gt 0) {
        Say "  row counts differ between runs or no rows; raw logs are in results\"
        return
    }
    for ($i = 0; $i -lt $n; $i++) {
        $rows = @($all | ForEach-Object { $_[$i] })
        $skel = @($rows | ForEach-Object { $TokenRe.Replace($_, "#") -replace '\s+', ' ' } | Select-Object -Unique)
        if ($skel.Count -ne 1) {
            Say ("  ${A}: " + $a[0][$i].Trim())
            Say ("  ${B}: " + $p[0][$i].Trim())
            continue
        }
        $tokens = @($rows | ForEach-Object { , @($TokenRe.Matches($_)) })
        $ctr = @{ k = 0 }
        $line = $TokenRe.Replace($rows[0], [Text.RegularExpressions.MatchEvaluator] {
            param($m)
            $j = $ctr.k; $ctr.k++
            $raw = @($tokens | ForEach-Object { $_[$j].Value } | Select-Object -Unique)
            if ($raw.Count -eq 1) { return $m.Value }
            $vals = @($tokens | ForEach-Object {
                ToNs ([double]::Parse($_[$j].Groups[1].Value, [Globalization.CultureInfo]::InvariantCulture)) $_[$j].Groups[2].Value
            })
            $va = Median $vals[0..($Rounds - 1)]
            $vp = Median $vals[$Rounds..(2 * $Rounds - 1)]
            $unit = $m.Groups[2].Value
            if ($unit -eq "x") { return "[$A $(Fmt $va)x | $B $(Fmt $vp)x]" }
            $suffix = ""
            if ($unit -ne "") { $suffix = "ns" }
            return "[$A $(Fmt $va)$suffix | $B $(Fmt $vp)$suffix | $(Fmt ($vp / $va))x]"
        })
        Say ("  " + $line.Trim())
    }
}

# --- System -------------------------------------------------------------------------------------

$cpu = Get-CimInstance Win32_Processor
Say "CPU: $($cpu.Name), $($cpu.NumberOfCores) cores, $($cpu.NumberOfLogicalProcessors) logical"
Say "OS: $([Environment]::OSVersion.VersionString)"
Say ("rustc: " + ((& rustc -vV) -join "; "))
Say ("power scheme: " + (& powercfg /getactivescheme))
$battery = Get-CimInstance Win32_Battery -ErrorAction SilentlyContinue
if ($battery -and $battery.BatteryStatus -ne 2) { Say "WARNING: running on battery; plug in for stable timings" }
if ($Affinity -eq "") { Say "timing runs not pinned, rounds: $Rounds" } else { Say "affinity mask for timing runs: 0x$Affinity, rounds: $Rounds" }
Say "Rows read [asm A | portable B | B/A x], medians over rounds; B/A above 1 means the asm build is faster."
Say "MULX dispatch rows read [cpuid A | folded B | B/A x]; B/A below 1 means the CPUID check costs time."
Say "[asm Ax | portable Bx] are the benchmark's own ratio columns, shown for both builds."

# --- Fork ---------------------------------------------------------------------------------------

$repo = Split-Path $PSScriptRoot -Parent
if (Test-Path (Join-Path $repo "curves/pallas/Cargo.toml")) {
    $fork = $repo
} else {
    $fork = Get-Source "https://github.com/PolymeshAssociation/arkworks-algebra/archive/refs/heads/$Branch.zip" "fork"
}
Say "code: $fork"
$pallas = "--manifest-path curves/pallas/Cargo.toml"

if (-not $SkipTests) {
    Say ""
    Say "== correctness (asm build, ARK_FF_REQUIRE_ASM=1)"
    $env:ARK_FF_REQUIRE_ASM = "1"
    foreach ($t in @(
            @{ Log = "test.ff.asm.txt"; Cmd = "cargo test --release --target-dir target-asm --manifest-path ff/Cargo.toml --features asm" },
            @{ Log = "test.pallas.asm.txt"; Cmd = "cargo test --release --target-dir target-asm $pallas --features asm" },
            @{ Log = "test.vesta.asm.txt"; Cmd = "cargo test --release --target-dir target-asm --manifest-path curves/vesta/Cargo.toml --features asm" })) {
        $code = Run $fork $t.Cmd $t.Log
        if ($code -eq 0) { Say "  pass  $($t.Log)" } else {
            Say "  FAIL  $($t.Log) (exit $code)"
            if ($t.Log -ne "test.ff.asm.txt") {
                $pl = $t.Log -replace "\.asm\.", ".portable."
                $pc = Run $fork (($t.Cmd -replace "target-asm", "target-portable") -replace " --features asm", "") $pl
                Say "        same tests without asm: exit $pc"
            }
        }
    }
    Remove-Item Env:ARK_FF_REQUIRE_ASM
}

$benches = @(
    @{ Name = "asm_backend_bench"; Extra = "" },
    @{ Name = "point_add_bench"; Extra = "" },
    @{ Name = "deferred_inner_product"; Extra = "" },
    @{ Name = "sqrt_bench"; Extra = "" },
    @{ Name = "glv_batch_affine_bench"; Extra = "parallel" })

function BenchCmd($b, [string]$Build, [string]$Tail) {
    $f = @()
    if ($Build -eq "asm") { $f += "asm" }
    if ($b.Extra -ne "") { $f += $b.Extra }
    $feat = ""
    if ($f.Count) { $feat = "--features " + ($f -join ",") }
    return "cargo test --release -q --target-dir target-$Build $pallas $feat --test $($b.Name) $Tail"
}

$env:MSM_SIZES = $MsmSizes
$env:MSM_THREADS = $Threads
Say ""
Say "Building benchmarks"
foreach ($b in $benches) {
    foreach ($build in @("asm", "portable")) {
        Run $fork (BenchCmd $b $build "--no-run") "build.$($b.Name).$build.txt" | Out-Null
    }
}
Run $fork "cargo test --release -q --target-dir target-asm $pallas --features asm --test loose_ops_bench --no-run" "build.loose_ops_bench.asm.txt" | Out-Null

Pin $true
for ($r = 1; $r -le $Rounds; $r++) {
    foreach ($b in $benches) {
        foreach ($build in @("asm", "portable")) {
            Write-Host "round $r/$Rounds  $($b.Name)  $build" -ForegroundColor Yellow
            Run $fork (BenchCmd $b $build "-- --ignored --nocapture") "$($b.Name).$build.$r.txt" | Out-Null
        }
    }
    Run $fork "cargo test --release -q --target-dir target-asm $pallas --features asm --test loose_ops_bench -- --ignored --nocapture" "loose_ops_bench.asm.$r.txt" | Out-Null
}
Pin $false

foreach ($b in $benches) { Compare-Builds $b.Name }
Say ""
Say "== loose_ops_bench (asm build; canonical against loose field operations, round 1)"
foreach ($l in (DataLines "loose_ops_bench.asm.1.txt")) { Say ("  " + $l.Trim()) }

# --- MULX/ADX selection --------------------------------------------------------------------------

# `cpuid` replaces .cargo/config.toml's `+bmi2,+adx` with a no-op flag, since Windows drops an empty
# environment variable, so the kernel is reached through the run-time CPUID check. `folded` keeps
# the config's flags, which fold the check away.
$dispatch = @{ cpuid = "-C debuginfo=0"; folded = $null }
$dispatchCurves = @("bls12_381", "bn254")

function Set-Dispatch([string]$Build) {
    if ($dispatch[$Build]) { $env:RUSTFLAGS = $dispatch[$Build] } else { Remove-Item Env:RUSTFLAGS -ErrorAction SilentlyContinue }
}

function DispatchCmd([string]$Curve, [string]$Build, [string]$Tail) {
    return "cargo test --release -q --target-dir target-$Build --manifest-path curves/$Curve/Cargo.toml --features asm --test mulx_dispatch_bench $Tail"
}

Say ""
Say "Building MULX dispatch benchmarks"
foreach ($c in $dispatchCurves) {
    foreach ($build in @("cpuid", "folded")) {
        Set-Dispatch $build
        Run $fork (DispatchCmd $c $build "--no-run") "build.mulx_dispatch_$c.$build.txt" | Out-Null
    }
}

Pin $true
for ($r = 1; $r -le $Rounds; $r++) {
    foreach ($c in $dispatchCurves) {
        foreach ($build in @("cpuid", "folded")) {
            Write-Host "round $r/$Rounds  mulx_dispatch $c  $build" -ForegroundColor Yellow
            Set-Dispatch $build
            Run $fork (DispatchCmd $c $build "-- --ignored --nocapture") "mulx_dispatch_$c.$build.$r.txt" | Out-Null
        }
    }
}
Pin $false
Remove-Item Env:RUSTFLAGS -ErrorAction SilentlyContinue

foreach ($c in $dispatchCurves) { Compare-Builds "mulx_dispatch_$c" "cpuid" "folded" }

# --- Zakura #701 --------------------------------------------------------------------------------

if ($Zakura) {
  try {
    $zk = Get-Source "https://github.com/zakura-core/common/archive/$ZakuraCommit.zip" "zakura"
    $flags = @{ scalar = "-C target-feature=+adx,+bmi2"; avx2 = "-C target-feature=+adx,+bmi2,+avx2" }
    $zcmd = "cargo +stable bench -p pasta_curves --features multicore,orbits --bench glv"
    Say ""
    Say "Building Zakura #701 benchmarks"
    foreach ($build in @("scalar", "avx2")) {
        $env:RUSTFLAGS = $flags[$build]
        Run $zk "$zcmd --target-dir target-$build --no-run" "build.zakura.$build.txt" | Out-Null
    }
    Pin $true
    foreach ($t in $Threads.Split(",")) {
        $env:RAYON_NUM_THREADS = $t
        for ($r = 1; $r -le $Rounds; $r++) {
            foreach ($build in @("scalar", "avx2")) {
                Write-Host "zakura threads $t round $r/$Rounds  $build" -ForegroundColor Yellow
                $env:RUSTFLAGS = $flags[$build]
                Run $zk "$zcmd --target-dir target-$build -- Pallas/multiexp" "zakura.t$t.$build.$r.txt" | Out-Null
            }
        }
    }
    Pin $false
    Remove-Item Env:RUSTFLAGS, Env:RAYON_NUM_THREADS

    # Criterion's median estimate per benchmark id.
    function CriterionTimes([string]$Log) {
        $times = @{}
        $id = $null
        $path = Join-Path $Out $Log
        if (-not (Test-Path $path)) { return $times }
        foreach ($l in Get-Content $path) {
            if ($l -match '^(Pallas/multiexp/\S+)') { $id = $Matches[1] }
            if ($id -and $l -match 'time:\s+\[\S+ \S+ (\d+(?:\.\d+)?) (\S+) ') {
                $times[$id] = ToNs ([double]::Parse($Matches[1], [Globalization.CultureInfo]::InvariantCulture)) $Matches[2]
                $id = $null
            }
        }
        return $times
    }

    Say ""
    Say "== Zakura #701 GLV MSM, Pallas   [scalar | avx2 | scalar/avx2], medians over rounds, ns"
    foreach ($t in $Threads.Split(",")) {
        Say "  threads $t"
        $runs = @{ scalar = @(); avx2 = @() }
        for ($r = 1; $r -le $Rounds; $r++) {
            foreach ($build in @("scalar", "avx2")) { $runs[$build] += , (CriterionTimes "zakura.t$t.$build.$r.txt") }
        }
        $ids = @($runs.scalar[0].Keys | Sort-Object { [int]($_ -replace '.*/', '') })
        if ($ids.Count -eq 0) { Say "    no results; see results\zakura.t$t.*.txt" }
        foreach ($id in $ids) {
            $s = Median @($runs.scalar | ForEach-Object { $_[$id] })
            $v = Median @($runs.avx2 | ForEach-Object { $_[$id] })
            Say "    $id  [$(Fmt $s) | $(Fmt $v) | $(Fmt ($s / $v))x]"
        }
    }
  } catch {
    Pin $false
    Say ""
    Say "Zakura step failed: $_"
  }
}

$zip = Join-Path $Root "x86-bench-results.zip"
Compress-Archive -Path (Join-Path $Out "*") -DestinationPath $zip -Force
Say ""
Say "Done. Send $zip"
