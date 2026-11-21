# Registers (or removes with -Uninstall) the per-user logon task that runs scripts/run_services.ps1.
param([switch]$Uninstall)

$TaskName = 'TradingAgents services'
if ($Uninstall) {
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
    Write-Output "removed: $TaskName"
    exit 0
}

$Script = Join-Path $PSScriptRoot 'run_services.ps1'
$Root = Split-Path -Parent $PSScriptRoot
$User = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name

$action = New-ScheduledTaskAction -Execute 'powershell.exe' -WorkingDirectory $Root `
    -Argument ('-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File "{0}"' -f $Script)
$trigger = New-ScheduledTaskTrigger -AtLogOn -User $User
$trigger.Delay = 'PT30S'  # let the network come up first
$settings = New-ScheduledTaskSettingsSet -ExecutionTimeLimit ([TimeSpan]::Zero) -MultipleInstances IgnoreNew `
    -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable `
    -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 1)
$principal = New-ScheduledTaskPrincipal -UserId $User -LogonType Interactive -RunLevel Limited

Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger -Settings $settings `
    -Principal $principal -Description 'Telegram collector, event agent and login gateway (C:\project\trading)' -Force | Out-Null
Write-Output "registered: $TaskName (at logon of $User)"
