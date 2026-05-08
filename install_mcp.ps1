#Requires -Version 5.1
<#
.SYNOPSIS
    Install the Chonk MCP server entry into your MCP host config.

.DESCRIPTION
    Writes the Chonk HTTP MCP server entry into the config file for your
    chosen MCP host. No Python or extra tools required.

.PARAMETER Url
    URL of the Chonk MCP HTTP server (required).

.PARAMETER ApiKey
    Bearer token. Required when the server has auth enabled.

.PARAMETER Name
    Entry name in the config (default: chonk).

.PARAMETER Host
    Target host config:
      claude           Claude Desktop (default)
      claude-code      Claude Code — project scope (.mcp.json in current dir)
      claude-code-user Claude Code — user scope (~/.claude.json)
      gemini           Google Gemini CLI
      copilot          GitHub Copilot in VS Code (.vscode/mcp.json)
      vscode           VS Code settings.json
      <file path>      Any JSON config file

.PARAMETER DryRun
    Print what would be written without modifying anything.

.EXAMPLE
    .\install_mcp.ps1 -Url http://chonk.internal:8000/mcp -ApiKey SECRET

.EXAMPLE
    .\install_mcp.ps1 -Url http://chonk.internal:8000/mcp -ApiKey SECRET -Host claude-code -DryRun
#>

[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)]
    [string]$Url,

    [string]$ApiKey = "",

    [string]$Name = "chonk",

    [string]$Host = "claude",

    [switch]$DryRun
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# ---------------------------------------------------------------------------
# Resolve config file path
# ---------------------------------------------------------------------------
$TypedHosts = @("claude-code", "claude-code-user", "copilot", "vscode")

switch ($Host) {
    "claude" {
        $ConfigPath = Join-Path $env:APPDATA "Claude\claude_desktop_config.json"
        $TopKey = "mcpServers"
    }
    "claude-code" {
        $ConfigPath = Join-Path (Get-Location) ".mcp.json"
        $TopKey = "mcpServers"
    }
    "claude-code-user" {
        $ConfigPath = Join-Path $HOME ".claude.json"
        $TopKey = "mcpServers"
    }
    "gemini" {
        $ConfigPath = Join-Path $HOME ".gemini\settings.json"
        $TopKey = "mcpServers"
    }
    "copilot" {
        $ConfigPath = Join-Path (Get-Location) ".vscode\mcp.json"
        $TopKey = "servers"
    }
    "vscode" {
        $ConfigPath = Join-Path $env:APPDATA "Code\User\settings.json"
        $TopKey = "mcp"   # handled specially below
    }
    default {
        $ConfigPath = $Host
        $TopKey = "mcpServers"
    }
}

# ---------------------------------------------------------------------------
# Build the entry object
# ---------------------------------------------------------------------------
$Entry = [ordered]@{}

if ($TypedHosts -contains $Host) {
    $Entry["type"] = "http"
}
$Entry["url"] = $Url
if ($ApiKey -ne "") {
    $Entry["headers"] = @{ Authorization = "Bearer $ApiKey" }
}

# ---------------------------------------------------------------------------
# Read existing config (or start fresh)
# ---------------------------------------------------------------------------
if (Test-Path $ConfigPath) {
    try {
        $Config = Get-Content $ConfigPath -Raw | ConvertFrom-Json -AsHashtable
    } catch {
        Write-Error "Cannot parse ${ConfigPath}: $_"
        exit 1
    }
} else {
    $Config = @{}
}

# ---------------------------------------------------------------------------
# Inject entry at correct key path
# ---------------------------------------------------------------------------
switch ($Host) {
    "vscode" {
        if (-not $Config.ContainsKey("mcp"))     { $Config["mcp"] = @{} }
        if (-not $Config["mcp"].ContainsKey("servers")) { $Config["mcp"]["servers"] = @{} }
        $Config["mcp"]["servers"][$Name] = $Entry
    }
    "copilot" {
        if (-not $Config.ContainsKey("servers")) { $Config["servers"] = @{} }
        $Config["servers"][$Name] = $Entry
    }
    default {
        if (-not $Config.ContainsKey("mcpServers")) { $Config["mcpServers"] = @{} }
        $Config["mcpServers"][$Name] = $Entry
    }
}

# ---------------------------------------------------------------------------
# Serialise
# ---------------------------------------------------------------------------
$NewJson = $Config | ConvertTo-Json -Depth 10

# ---------------------------------------------------------------------------
# Write or print
# ---------------------------------------------------------------------------
if ($DryRun) {
    Write-Host "[dry-run] Would write to: $ConfigPath"
    Write-Host $NewJson
    exit 0
}

$Dir = Split-Path $ConfigPath -Parent
if (-not (Test-Path $Dir)) {
    New-Item -ItemType Directory -Path $Dir -Force | Out-Null
}

Set-Content -Path $ConfigPath -Value $NewJson -Encoding UTF8

Write-Host "Installed '$Name' -> $ConfigPath"
if ($Host -in @("claude-code", "claude-code-user")) {
    Write-Host "Run: claude mcp list"
} else {
    Write-Host "Restart your MCP host to pick up the change."
}
