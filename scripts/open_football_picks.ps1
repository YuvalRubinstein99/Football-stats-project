param([switch]$NoBrowser)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$appUrl = 'http://127.0.0.1:8767/'
$pythonPath = Join-Path $env:USERPROFILE '.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
$logDirectory = Join-Path $projectRoot 'results\model_strategy'
$mutex = New-Object System.Threading.Mutex($false, 'Local\FootballPicksLauncher')
$locked = $false

function Get-AppPage {
    $request = [System.Net.HttpWebRequest]::Create($appUrl)
    $request.Proxy = $null
    $request.Timeout = 2000
    try {
        $response = $request.GetResponse()
        $reader = New-Object System.IO.StreamReader($response.GetResponseStream())
        try { return $reader.ReadToEnd() }
        finally { $reader.Dispose(); $response.Dispose() }
    }
    catch [System.Net.WebException] { return $null }
}

try {
    $locked = $mutex.WaitOne(60000)
    if (-not $locked) { throw 'Another launch is still starting. Try again shortly.' }
    $page = Get-AppPage
    if ($page -and -not $page.Contains('Upcoming outcome picks')) {
        throw 'Port 8767 is serving another page or an older app. Close that server and try again.'
    }
    if (-not $page) {
        if (-not (Test-Path -LiteralPath $pythonPath)) {
            throw "Python runtime missing at $pythonPath"
        }
        New-Item -ItemType Directory -Force -Path $logDirectory | Out-Null
        $options = @{
            FilePath = $pythonPath
            ArgumentList = '-u run.py model-app'
            WorkingDirectory = $projectRoot
            WindowStyle = 'Hidden'
            PassThru = $true
            RedirectStandardOutput = Join-Path $logDirectory 'app.stdout.log'
            RedirectStandardError = Join-Path $logDirectory 'app.stderr.log'
        }
        $server = Start-Process @options
        $deadline = (Get-Date).AddSeconds(60)
        do {
            Start-Sleep -Milliseconds 400
            $page = Get-AppPage
            if ($server.HasExited -and -not $page) {
                throw "App could not start. See $logDirectory\app.stderr.log"
            }
        } until ($page -or (Get-Date) -gt $deadline)
        if (-not $page -or -not $page.Contains('Upcoming outcome picks')) {
            throw "App did not become ready. See $logDirectory\app.stderr.log"
        }
    }
    if (-not $NoBrowser) { Start-Process $appUrl }
    Write-Output "Football Picks is ready at $appUrl"
}
finally {
    if ($locked) { $mutex.ReleaseMutex() }
    $mutex.Dispose()
}
