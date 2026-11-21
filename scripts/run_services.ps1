# Keeps the Telegram collector, the event-analysis agent and the login gateway running.
# Started at logon by the "TradingAgents services" scheduled task (see scripts/install_autostart.ps1).
# Every minute it starts any service that is not running; logs go to data/logs/.

$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
$Py = Join-Path $Root '.venv\Scripts\python.exe'
$LogDir = Join-Path $Root 'data\logs'
New-Item -ItemType Directory -Force $LogDir | Out-Null
$SupervisorLog = Join-Path $LogDir 'supervisor.log'

function Write-Log([string]$msg) {
    $line = '{0} {1}' -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $msg
    Add-Content -Path $SupervisorLog -Value $line -Encoding UTF8
}

# Only one supervisor at a time.
$created = $false
$mutex = New-Object System.Threading.Mutex($true, 'Local\TradingAgentsServices', [ref]$created)
if (-not $created) { exit 0 }

$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONUNBUFFERED = '1'

$Services = @(
    @{ Name = 'telegram';    Script = 'telegram_collector.py'; Verb = 'run' },
    @{ Name = 'event-agent'; Script = 'news_event_agent.py';   Verb = 'run' },
    @{ Name = 'gateway';     Script = 'auth_gateway.py';       Verb = 'serve' }
)

function Test-Running($svc) {
    $pattern = '*{0}*{1}*' -f $svc.Script, $svc.Verb
    $procs = Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
        Where-Object { $_.CommandLine -like $pattern }
    return [bool]$procs
}

function Start-Service2($svc) {
    $out = Join-Path $LogDir ($svc.Name + '.log')
    $err = Join-Path $LogDir ($svc.Name + '.err.log')
    foreach ($f in @($out, $err)) {
        if (Test-Path $f) { Move-Item -Force $f ($f + '.prev') }
    }
    Start-Process -FilePath $Py -ArgumentList @($svc.Script, $svc.Verb) -WorkingDirectory $Root `
        -WindowStyle Hidden -RedirectStandardOutput $out -RedirectStandardError $err | Out-Null
    Write-Log ('started ' + $svc.Name)
}

Write-Log 'supervisor started'
try {
    while ($true) {
        foreach ($svc in $Services) {
            try {
                if (-not (Test-Running $svc)) { Start-Service2 $svc }
            } catch {
                Write-Log ('failed to start {0}: {1}' -f $svc.Name, $_.Exception.Message)
            }
        }
        Start-Sleep -Seconds 60
    }
} finally {
    Write-Log 'supervisor stopped'
    $mutex.ReleaseMutex()
}
