// @ts-nocheck
/**
 * File: src/tools/FileReadTool/prompt.js
 * Role: UTILITY_HELPER
 * Aggregated from 11 chunks
 */

import { resolve as resolveAbsolutePath, join as joinPath, extname as getExtension, basename as getBaseName } from "path";
const pathJoin = joinPath; // Alias for compatibility
import { statSync, readFileSync, mkdirSync, writeFileSync, unlinkSync, existsSync } from "fs";
import { fileURLToPath } from "url";
import * as os from "os";

// Constants
const SETTING_LAYERS = ["flagSettings", "policySettings", "userSettings", "projectSettings", "localSettings"];

// Stub functions for missing dependencies (would be imported from a config service)
function getSettings() { return {}; }
function getToolSettings(layer) { return {}; }
function setToolSettings(layer, settings) { }
function getRipgrepSettings() { return { rgPath: 'rg', rgArgs: [] }; }
function isSandboxPolicyEnabled(settings) { return false; }
function isSandboxConfigured(settings) { return false; }
function isSandboxSupported() { return true; }
function isSessionActive() { return true; }
function getPlatform() { return process.platform; }
function getUserHomeDirectory() { return os.homedir(); }
function getProjectRoot() { return process.cwd(); } // Mock
function getPolicyPath() { return "/etc/claude"; } // Mock
function isBuiltInAgent(agent) { return agent.source === "built-in"; }
function isPluginAgent(agent) { return agent.source === "plugin"; }
function getFileSystem() { return { existsSync, mkdirSync, unlinkSync, writeFileSync, readFileSync }; }
function writeFile(path, content, options) { writeFileSync(path, content, options); }

// React stubs
const React = { Fragment: ({ children }) => children, useRef: () => ({ current: null }), useState: (v) => [v, () => { }] };
const useMemo = (fn) => fn();
const I = ({ children }) => children;
const f = ({ children }) => children;

const memoize = (fn) => { const cache = new Map(); const m = (...args) => { const k = args[0]; if (cache.has(k)) return cache.get(k); const v = fn(...args); cache.set(k, v); return v; }; m.cache = cache; return m; };
// Variables declared later: allAvailableCommands, commandNameSet, getFilteredAndSortedCommands

const isProductionBuild = () => false;

// Dummy Initializers
function initializeProjectCommands() { }
function initializeRemoteCommands() { }
function initializeSettingsCommands() { }
function initializeHelpCommands() { }
function initializeAgentsCommands() { }
function initializeUtilityCommands() { }
function initializeIntegrations() { }
function initializePluginsCommands() { }
function initializeFileCache() { }
function initializeCompletionProviders() { }
function initializeUsageReports() { }
function initializeReleaseNotes() { }
function initializePlatform() { }
function initializeLsp() { }
function initializeNetworkStatus() { }
function initializeNotifications() { }
function initializeVimMode() { }
function initializeProduct() { }
function initializeGlobalStyles() { }
function initializeReactComponents() { }
function getReactLibrary() { return React; }
function initializeEnvironmentVariables() { }
function initializeCommandLineFlags() { }
function initializeErrorBoundary() { }
function initializeSlashCommandDispatcher() { }
function initializeTelemetry() { }
function initializeLocalization() { }
function initializeHotkeys() { }
function configureTheming() { }
function registerServices() { }
function initializeToolRegistry() { }
function initializeConstants() { }
function initializeCommandFlags() { }
function initializeAutocompleteCommands() { }
function initializeTerminalCommands() { }
function initializeVcsCommands() { }
function initializeAuthCommands() { }
const C = (fn) => fn();


// --- Chunk: chunk252 (Original lines: 103903-104119) ---
/**
 * Function to parse a tool permission rule.
 * @param {string} rule - The permission rule string.
 * @returns {object} - An object containing the tool name and optional rule content.
 */
function parseToolPermissionRule(rule) {
  const match = rule.match(/^([^(]+)\(([^)]+)\)$/);

  if (!match) {
    return {
      toolName: rule
    };
  }

  const toolName = match[1];
  const ruleContent = match[2];

  if (!toolName || !ruleContent) {
    return {
      toolName: rule
    };
  }

  return {
    toolName,
    ruleContent
  };
}

/**
 * Extracts the domain from a wildcard string.
 * @param {string} input - Input string.
 * @returns {string|null} - Domain or null if not found.
 */
function extractDomain(input) {
  const match = input.match(/^(.+):\*$/);
  return match?.[1] ?? null;
}

/**
 * Resolves the absolute path for a tool.
 * @param {string} toolPath - The original tool path.
 * @param {string} contextPath - The context path.
 * @returns {string} - The resolved tool path.
 */
function resolveToolPath(toolPath, contextPath) {
  if (toolPath.startsWith("//")) {
    return toolPath.slice(1);
  }

  if (toolPath.startsWith("/") && !toolPath.startsWith("//")) {
    const claudePath = getClaudePath(contextPath);
    return resolveAbsolutePath(claudePath, toolPath.slice(1));
  }

  return toolPath;
}

/**
 * Gets the network, filesystem, and ripgrep settings
 * based on the provided tool configuration.
 * @param {object} config - The tool configuration object.
 * @returns {object} - An object containing the network, filesystem,
 *  and ripgrep settings.
 */
function getSandboxSettings(config) {
  const permissions = config.permissions || {};
  const allowedDomains = [];
  const deniedDomains = [];

  // Add allowed domains from sandbox network settings
  for (const domain of config.sandbox?.network?.allowedDomains || []) {
    allowedDomains.push(domain);
  }

  // Add allowed domains from 'allow' permissions
  for (const allowRule of permissions.allow || []) {
    const parsedRule = parseToolPermissionRule(allowRule);
    if (parsedRule.toolName === 'file_read' && parsedRule.ruleContent?.startsWith("domain:")) {
      allowedDomains.push(parsedRule.ruleContent.substring(7));
    }
  }

  // Add denied domains from 'deny' permissions
  for (const denyRule of permissions.deny || []) {
    const parsedRule = parseToolPermissionRule(denyRule);
    if (parsedRule.toolName === 'file_read' && parsedRule.ruleContent?.startsWith("domain:")) {
      deniedDomains.push(parsedRule.ruleContent.substring(7));
    }
  }

  const allowWritePaths = ["."];
  const denyWritePaths = [];
  const filesDenyWritePaths = [];

  const claudePath = getClaudePath(config); // Assuming getClaudePath is defined elsewhere
  const gitPath = joinPath(claudePath, ".git");

  try {
    if (statSync(gitPath).isFile()) {
      const gitDirMatch = readFileSync(gitPath, {
        encoding: "utf8"
      }).match(/^gitdir:\s*(.+)$/m);
      if (gitDirMatch?.[1]) {
        const gitDir = gitDirMatch[1].trim();
        const gitDirIndex = gitDir.indexOf(".git");
        if (gitDirIndex > 0) {
          const gitRoot = gitDir.substring(0, gitDirIndex - 1);
          if (gitRoot !== claudePath) {
            allowWritePaths.push(gitRoot);
          }
        }
      }
    }
  } catch { }

  for (const setting of SETTING_LAYERS) {
    const toolSettings = getToolSettings(setting);
    if (!toolSettings?.permissions) {
      continue;
    }

    for (const allow of toolSettings.permissions.allow || []) {
      const parsedAllowRule = parseToolPermissionRule(allow);
      if (parsedAllowRule.toolName === 'file_write' && parsedAllowRule.ruleContent) {
        allowWritePaths.push(resolveToolPath(parsedAllowRule.ruleContent, setting));
      }
    }

    for (const deny of toolSettings.permissions.deny || []) {
      const parsedDenyRule = parseToolPermissionRule(deny);
      if (parsedDenyRule.toolName === 'file_write' && parsedDenyRule.ruleContent) {
        denyWritePaths.push(resolveToolPath(parsedDenyRule.ruleContent, setting));
      }
      if (parsedDenyRule.toolName === 'file_read' && parsedDenyRule.ruleContent) {
        filesDenyWritePaths.push(resolveToolPath(parsedDenyRule.ruleContent, setting));
      }
    }
  }


  const ripgrepSettings = config.sandbox?.ripgrep || (() => {
    const {
      rgPath,
      rgArgs
    } = getRipgrepSettings(); // Assuming RzA() is defined elsewhere
    return {
      command: rgPath,
      args: rgArgs
    };
  })();

  return {
    network: {
      allowedDomains,
      deniedDomains,
      allowUnixSockets: config.sandbox?.network?.allowUnixSockets,
      allowAllUnixSockets: config.sandbox?.network?.allowAllUnixSockets,
      allowLocalBinding: config.sandbox?.network?.allowLocalBinding,
      httpProxyPort: config.sandbox?.network?.httpProxyPort,
      socksProxyPort: config.sandbox?.network?.socksProxyPort
    },
    filesystem: {
      denyRead: filesDenyWritePaths,
      allowWrite: allowWritePaths,
      denyWrite: denyWritePaths
    },
    ignoreViolations: config.sandbox?.ignoreViolations,
    enableWeakerNestedSandbox: config.sandbox?.enableWeakerNestedSandbox,
    ripgrep: ripgrepSettings
  };
}

/**
 * Checks if policy settings are enabled.
 * @returns {boolean} - True if enabled, false otherwise.
 */
function isSettingsSandboxEnabled() {
  try {
    const settings = getSettings(); // Assuming G8 is defined elsewhere
    return isSandboxPolicyEnabled(settings);
  } catch (error) {
    console.error(`Failed to get settings for sandbox check: ${error}`);
    return false;
  }
}

/**
 * Checks if the sandbox is enabled based on the settings configuration.
 * @returns {boolean} - True if enabled, false otherwise.
 */
function isSandboxEnabledInSettings() {
  const settings = getSettings();
  return isSandboxConfigured(settings);
}

/**
 * Checks if sandbox is enabled.
 * @returns {boolean} - True if both settings and policy enable sandbox.
 */
function isSandboxActive() {
  if (!isSandboxSupported()) {
    return false;
  }
  if (!isSessionActive()) {
    return false;
  }
  return isSettingsSandboxEnabled();
}

/**
 * Gets warnings for Linux glob patterns.
 * @returns {string[]} - An array of glob patterns.
 */
function getLinuxGlobPatternWarnings() {
  const platform = getPlatform(); // Assuming q8 is defined elsewhere
  if (platform !== "linux" && platform !== "wsl") {
    return [];
  }

  try {
    const settings = getSettings();
    if (!settings?.sandbox?.enabled) {
      return [];
    }

    const {
      permissions = {}
    } = settings;
    const warnings = [];

    const isGlobPattern = (path) => {
      const noWildcard = path.replace(/\/\*\*$/, "");
      return /[*?[\]]/.test(noWildcard);
    };


    for (const permission of [...(permissions.allow || []), ...(permissions.deny || [])]) {
      const parsedPermission = parseToolPermissionRule(permission);
      if ((parsedPermission.toolName === 'file_write' || parsedPermission.toolName === 'file_read') && parsedPermission.ruleContent && isGlobPattern(parsedPermission.ruleContent)) {
        warnings.push(permission);
      }
    }

    return warnings;
  } catch (error) {
    console.error(`Failed to get Linux glob pattern warnings: ${error}`);
    return [];
  }
}

/**
 * Checks if sandbox setting is present in flagSettings or policySettings
 * @returns {boolean}
 */
function hasSandboxSettings() {
  const settingsTypes = ["flagSettings", "policySettings"];
  for (const settingsType of settingsTypes) {
    const settings = getToolSettings(settingsType); // Assuming v4 is defined elsewhere
    if (settings?.sandbox?.enabled !== void 0 || settings?.sandbox?.autoAllowBashIfSandboxed !== void 0 || settings?.sandbox?.allowUnsandboxedCommands !== void 0) {
      return true;
    }
  }
  return false;
}

/**
 * Updates the local sandbox settings.
 * @param {object} settings - The new sandbox settings.
 */
async function updateLocalSandboxSettings(settings) {
  const localSettings = getToolSettings("localSettings"); // Assuming v4 is defined elsewhere
  setToolSettings("localSettings", { // Assuming I4 is defined elsewhere
    sandbox: {
      ...localSettings?.sandbox,
      ...(settings.enabled !== void 0 && {
        enabled: settings.enabled
      }),
      ...(settings.autoAllowBashIfSandboxed !== void 0 && {
        autoAllowBashIfSandboxed: settings.autoAllowBashIfSandboxed
      }),
      ...(settings.allowUnsandboxedCommands !== void 0 && {
        allowUnsandboxedCommands: settings.allowUnsandboxedCommands
      })
    }
  });
}

/**
 * Gets excluded commands from the sandbox settings.
 * @returns {string[]} - An array of excluded command names.
 */
function getSandboxExcludedCommands() {
  return getSettings()?.sandbox?.excludedCommands ?? []; // Assuming G8 is defined elsewhere
}

// --- Chunk: chunk1390 (Original lines: 541266-541510) ---

import { join as pathJoin } from "path";

/**
 * Formats agent markdown content.
 * @param {string} name - Agent name.
 * @param {string} description - Agent description.
 * @param {string[]} tools - Tools used by the agent.
 * @param {string} content - Markdown content of the agent.
 * @param {string} color - Color for the agent.
 * @param {string} model - Model used by the agent.
 * @param {string} memory - Memory setting for the agent.
 * @returns {string} - Formatted agent markdown.
 */
function formatAgentMarkdown(name, description, tools = [], content, color, model, memory) {
  const escapedDescription = description.replace(/\\/g, "\\\\").replace(/"/g, "\\\"").replace(/\n/g, "\\\\n");

  const toolsSection = (tools === undefined || tools.length === 0 || (tools.length === 1 && tools[0] === "*")) ? "" : `
tools: ${tools.join(", ")}`;

  const modelSection = model ? `
model: ${model}` : "";
  const colorSection = color ? `
color: ${color}` : "";
  const memorySection = memory ? `
memory: ${memory}` : "";

  return `---
name: ${name}
description: "${escapedDescription}"${toolsSection}${modelSection}${colorSection}${memorySection}
---

${content}
`;
}

/**
 * Gets the agent directory path based on the source.
 * @param {string} source - The source of the agent.
 * @returns {string} - The agent directory path.
 */
function getAgentDirectoryPath(source) {
  switch (source) {
    case "flagSettings":
      throw new Error(`Cannot get directory path for ${source} agents`);
    case "userSettings":
      return pathJoin(getUserHomeDirectory(), ".claude", "agents");
    case "projectSettings":
      return pathJoin(getProjectRoot(), ".claude", "agents");
    case "policySettings":
      return pathJoin(getPolicyPath(), ".claude", "agents");
    case "localSettings":
      return pathJoin(getProjectRoot(), ".claude", "agents");
    default:
      return "";
  }
}

/**
 * Gets the agent file path for a source.
 * @param {string} source - Agent source.
 * @returns {string} - The agent file path.
 */
function getAgentFilePathForSource(source) {
  switch (source) {
    case "projectSettings":
      return pathJoin(".", ".claude", "agents");
    default:
      return getAgentDirectoryPath(source);
  }
}


/**
 * Constructs the file path for saving an agent file.
 * @param {object} agent - Agent details including source and agentType.
 * @returns {string} - The file path.
 */
function getAgentFilePathForSave(agent) {
  if (agent.source === "built-in") {
    return "Built-in";
  }

  if (agent.source === "plugin") {
    throw new Error("Cannot get file path for plugin agents");
  }

  const directoryPath = getAgentDirectoryPath(agent.source);
  const fileName = agent.filename || agent.agentType;
  return pathJoin(directoryPath, `${fileName}.md`);
}

/**
 * Constructs agent file path for settings.
 * @param {object} agent - Agent details including source and agentType.
 * @returns {string} - The file path.
 */
function getAgentFilePathForSettings(agent) {
  if (agent.source === "built-in") {
    return "Built-in";
  }

  const sourcePath = getAgentFilePathForSource(agent.source);
  return pathJoin(sourcePath, `${agent.agentType}.md`);
}

/**
 * Gets the full file path for an agent based on its source.
 * @param {object} agent - Agent details.
 * @returns {string} - The agent file path.
 */
function getAgentFilePath(agent) {
  if (isBuiltInAgent(agent)) {
    return "Built-in";
  }

  if (isPluginAgent(agent)) {
    return `Plugin: ${agent.plugin || "Unknown"}`;
  }

  const sourcePath = getAgentFilePathForSource(agent.source);
  const fileName = agent.filename || agent.agentType;
  return pathJoin(sourcePath, `${fileName}.md`);
}

/**
 * Creates the agent directory if it doesn't already exist.
 * @param {string} agentPath - The path to the agent directory.
 * @returns {string} - The agent directory path.
 */
function ensureAgentDirectoryExists(agentPath) {
  const fileSystem = getFileSystem(); // Assuming hA() is defined elsewhere
  if (!fileSystem.existsSync(agentPath)) {
    fileSystem.mkdirSync(agentPath, {
      recursive: true // Create parent directories if they don't exist
    });
  }
  return agentPath;
}

/**
 * Saves an agent to a file.
 * @param {string} source - The source of the agent.
 * @param {string} agentType - The type of the agent.
 * @param {string} description - The agent description.
 * @param {string[]} tools - The tools used by the agent.
 * @param {string} content - The markdown content.
 * @param {boolean} [overwrite=true] - Whether to overwrite an existing file.
 * @param {string} [color] - The color of the agent.
 * @param {string} [model] - The model used by the agent.
 * @param {string} [memory] - The memory setting for the agent.
 */
async function saveAgentToFile(source, agentType, description, tools, content, overwrite = true, color, model, memory) {
  if (source === "built-in") {
    throw new Error("Cannot save built-in agents");
  }

  const agentDirectory = getAgentDirectoryPath(source);
  ensureAgentDirectoryExists(agentDirectory);
  const filePath = getAgentFilePathForSave({
    source,
    agentType,
  });


  const fileSystem = getFileSystem(); // Assuming hA() is defined elsewhere
  if (overwrite && fileSystem.existsSync(filePath)) {
    throw new Error(`Agent file already exists: ${filePath}`);
  }

  const markdownContent = formatAgentMarkdown(agentType, description, tools, content, color, model, memory);
  writeFile(filePath, markdownContent, {
    encoding: "utf-8",
    flush: true
  });
}

/**
 * Updates an agent file.
 * @param {object} agent - Agent details.
 * @param {string} description - The agent description.
 * @param {string[]} tools - The tools used by the agent.
 * @param {string} content - Markdown content.
 * @param {string} [color] - The color of the agent.
 * @param {string} [model] - The model used by the agent.
 * @param {string} [memory] - The memory setting of the agent.
 */
async function updateAgentFile(agent, description, tools, content, color, model, memory) {
  if (agent.source === "built-in") {
    throw new Error("Cannot update built-in agents");
  }

  const filePath = getAgentFilePathForSave(agent);
  const markdownContent = formatAgentMarkdown(agent.agentType, description, tools, content, color, model, memory);

  writeFile(filePath, markdownContent, {
    encoding: "utf-8",
    flush: true
  });
}

/**
 * Deletes an agent file.
 * @param {object} agent - The agent details.
 */
async function deleteAgentFile(agent) {
  if (agent.source === "built-in") {
    throw new Error("Cannot delete built-in agents");
  }

  const fileSystem = getFileSystem(); // Assuming hA() is defined elsewhere
  const filePath = getAgentFilePathForSave(agent);
  if (fileSystem.existsSync(filePath)) {
    fileSystem.unlinkSync(filePath);
  }
}

// mMA is a function that appears to triggers some side-effects.
// It's likely related to initialization or setup and is intentionally included.
var mMA = (() => {
  initializeReact(); // Assuming e6 is defined elsewhere
  configureTheming(); // Assuming g4 is defined elsewhere
  initializeLocalization(); // Assuming F1 is defined elsewhere
  initializeHotkeys(); // Assuming r6 is defined elsewhere
  initializeErrorBoundary(); // Assuming DC is defined elsewhere
  initializeSlashCommandDispatcher(); // Assuming CZK is defined elsewhere
  initializeTelemetry(); // Assuming u8 is defined elsewhere
});



/**
 * Renders the content of an agent card.
 * @param {object} props - Props for the agent card.
 * @returns {JSX.Element} - The rendered agent card content.
 */
function renderAgentCardContent(props) {
  const memoizationCache = useMemo(() => [], []); // Use useMemo for memoization (React)
  const {
    title,
    titleColor = "text",
    subtitle,
    borderColor = "suggestion",
    borderDimColor,
    children,
    footer,
    titleSuffix
  } = props;

  let titleElement;
  if (memoizationCache[0] !== subtitle || memoizationCache[1] !== title || memoizationCache[2] !== titleColor || memoizationCache[3] !== titleSuffix) {
    titleElement = (
      <React.Fragment> {/* Use React.Fragment to avoid unnecessary DOM nodes */}
        {title && (
          <I flexDirection="column" paddingX={1}>
            <f bold color={titleColor}>{title}{titleSuffix}</f>
            {subtitle && <f dimColor>{subtitle}</f>}
          </I>
        )}
      </React.Fragment>
    );

    memoizationCache[0] = subtitle;
    memoizationCache[1] = title;
    memoizationCache[2] = titleColor;
    memoizationCache[3] = titleSuffix;
    memoizationCache[4] = titleElement;
  } else {
    titleElement = memoizationCache[4];
  }

  let childrenElement;
  if (memoizationCache[5] !== children) {
    childrenElement = (
      <I paddingX={1} flexDirection="column">
        {children}
      </I>
    );
    memoizationCache[5] = children;
    memoizationCache[6] = childrenElement;
  } else {
    childrenElement = memoizationCache[6];
  }

  let cardContent;
  if (memoizationCache[7] !== borderColor || memoizationCache[8] !== borderDimColor || memoizationCache[9] !== titleElement || memoizationCache[10] !== childrenElement) {
    cardContent = (
      <I borderStyle="round" borderColor={borderColor} borderDimColor={borderDimColor} flexDirection="column">
        {titleElement}
        {childrenElement}
      </I>
    );

    memoizationCache[7] = borderColor;
    memoizationCache[8] = borderDimColor;
    memoizationCache[9] = titleElement;
    memoizationCache[10] = childrenElement;
    memoizationCache[11] = cardContent;
  } else {
    cardContent = memoizationCache[11];
  }

  let finalElement;
  if (memoizationCache[12] !== footer || memoizationCache[13] !== cardContent) {
    finalElement = (
      <React.Fragment>
        {cardContent}
        {footer}
      </React.Fragment>
    );
    memoizationCache[12] = footer;
    memoizationCache[13] = cardContent;
    memoizationCache[14] = finalElement;
  } else {
    finalElement = memoizationCache[14];
  }

  return finalElement;
}


// tB6 is a function that triggers some side effects
// var React; // Removed duplicate declaration

var tB6 = (() => {
  initializeGlobalStyles(); // Assuming lA() is defined elsewhere
  initializeReactComponents(); // Assuming uA() is defined elsewhere
  // React = getReactLibrary(); // Removed re-assignment
});

/**
 * Renders a card component.
 * @param {object} props - Props for the card.
 * @returns {JSX.Element} - The rendered card.
 */
function renderCard(props) {

  const memoizationCache = useMemo(() => [], []); // Use useMemo for memoization (React)
  const {
    title,
    titleColor = "text",
    borderColor = "suggestion",
    children,
    subtitle
  } = props;
  let cardElement;

  if (memoizationCache[0] !== borderColor || memoizationCache[1] !== children || memoizationCache[2] !== subtitle || memoizationCache[3] !== title || memoizationCache[4] !== titleColor) {
    cardElement = (
      <renderAgentCardContent
        title={title}
        titleColor={titleColor}
        borderColor={borderColor}
        subtitle={subtitle}
      >
        {children}
      </renderAgentCardContent>
    );

    memoizationCache[0] = borderColor;
    memoizationCache[1] = children;
    memoizationCache[2] = subtitle;
    memoizationCache[3] = title;
    memoizationCache[4] = titleColor;
    memoizationCache[5] = cardElement;
  } else {
    cardElement = memoizationCache[5];
  }
  return cardElement;
}

// Au6 is a function that triggers some side effects.
var eB6;
var Au6 = (() => {
  initializeGlobalStyles(); // Assuming lA() is defined elsewhere
  tB6();
  eB6 = getReactLibrary(); // Assumes r(JA(), 1) is a function and returns the react library
});

// --- Chunk: chunk1451 (Original lines: 553552-553785) ---
// import { join as pathJoin, extname as getExtension, basename as getBaseName } from "path"; // Removed duplicate import

var usageDataDirectoryPath;
var usageDataFacetsPath;

// Initialization function for usage data paths
var initializeUsageDataPaths = (() => {
  initializeEnvironmentVariables(); // Assuming X5 is defined elsewhere
  initializeCommandLineFlags(); // Assuming J7 is defined elsewhere
  initializeReact(); // Assuming e6 is defined elsewhere
  initializeLocalization(); // Assuming F1 is defined elsewhere
  initializeHotkeys(); // Assuming r6 is defined elsewhere
  initializeErrorBoundary(); // Assuming I1 is defined elsewhere
  initializeNetworkStatus(); // Assuming Nw is defined elsewhere
  initializeLocalization(); // Assuming F1 is defined elsewhere
  usageDataDirectoryPath = pathJoin(getUserHomeDirectory(), "usage-data"); // Assuming f8 is defined elsewhere
  usageDataFacetsPath = pathJoin(usageDataDirectoryPath, "facets");
});

var fileSizeBytes;

// Initialization function for project setup
var initializeProjectSetup = (() => {
  initializePlatform(); // Assuming U4 is defined elsewhere
  initializeTelemetry(); // Assuming WT is defined elsewhere
  initializeHotkeys(); // Assuming r6 is defined elsewhere
  fileSizeBytes = getFileSizeProvider(); // Assuming r(Ch(), 1) is a function that returns the file size, Ch() is likely a function.
});


/**
 * Gets eligible commands for the current context.
 * @returns {Promise<string[]>} - A promise that resolves to an array of command names.
 */
async function getEligibleCommandsForCurrentContext() {
  try {
    const context = await getCurrentContext(); // Assuming nUA() is defined elsewhere
    return context?.eligible ? [D_K] : []; // Assuming D_K is defined elsewhere
  } catch (error) {
    return [];
  }
}

/**
 * Gets skill commands.
 * @param {string} filter - Filter string.
 * @returns {Promise<object>} - A promise that resolves to an object containing skill commands.
 */
async function getSkillCommands(filter) {
  try {
    const [skillDirCommands, pluginSkills, bundledSkills] = await Promise.all([
      getSkillDirectoryCommands(filter).catch(error => {
        console.error(error instanceof Error ? error : Error("Failed to load skill directory commands"));
        console.warn("Skill directory commands failed to load, continuing without them");
        return [];
      }),
      getPluginSkills().catch(error => {
        console.error(error instanceof Error ? error : Error("Failed to load plugin skills"));
        console.warn("Plugin skills failed to load, continuing without them");
        return [];
      }),
      // No changes needed
    ]);

    console.log(`getSkills returning: ${skillDirCommands.length} skill dir commands, ${pluginSkills.length} plugin skills, ${bundledSkills.length} bundled skills`);
    return {
      skillDirCommands,
      pluginSkills,
      bundledSkills
    };
  } catch (error) {
    console.error(error instanceof Error ? error : Error("Unexpected error loading skills"));
    console.warn("Unexpected error in getSkills, returning empty");
    return {
      skillDirCommands: [],
      pluginSkills: [],
      bundledSkills: []
    };
  }
}

/**
 * Sets up the project file watcher.
 */
function setupProjectFileWatcher() {
  // Clear caches to refresh command definitions.
  if (getFilteredAndSortedCommands.cache?.clear) {
    getFilteredAndSortedCommands.cache.clear();
  }
  if (getPromptSkillMetadata.cache?.clear) {
    getPromptSkillMetadata.cache.clear();
  }
  if (getFilteredSlashCommandSkills.cache?.clear) {
    getFilteredSlashCommandSkills.cache.clear();
  }
}

/**
 * Refreshes tools, reloading definitions and configurations.
 */
function refreshTools() {
  setupProjectFileWatcher();
  initializeToolRegistry(); // Assuming IP1 is defined elsewhere
  initializeLsp(); // Assuming K3K is defined elsewhere
  initializeFileCache(); // Assuming WD1 is defined elsewhere
}

/**
 * Checks if a command name or alias exists in a list of commands.
 * @param {string} commandName - The command name to check.
 * @param {object[]} commands - An array of command objects.
 * @returns {boolean} - True if the command or alias exists, false otherwise.
 */
function commandExists(commandName, commands) {
  return commands.some(command => command.name === commandName || command.userFacingName() === commandName || (command.aliases && command.aliases.includes(commandName)));
}

/**
 * Finds a command by its name or alias.
 * @param {string} commandName - The name or alias of the command.
 * @param {object[]} commands - An array of command objects.
 * @returns {object} - The found command object.
 * @throws {ReferenceError} - If the command is not found.
 */
function findCommandByName(commandName, commands) {
  const command = commands.find(cmd => cmd.name === commandName || cmd.userFacingName() === commandName || (cmd.aliases && cmd.aliases.includes(commandName)));

  if (!command) {
    const availableCommands = commands.map(cmd => {
      const userFacingName = cmd.userFacingName();
      return cmd.aliases ? `${userFacingName} (aliases: ${cmd.aliases.join(", ")})` : userFacingName;
    }).sort((a, b) => a.localeCompare(b)).join(", ");
    throw new ReferenceError(`Command ${commandName} not found. Available commands: ${availableCommands}`);
  }

  return command;
}

/**
 * Gets the description of a command.
 * @param {object} command - The command object.
 * @returns {string} - The command description.
 */
function getCommandDescription(command) {
  if (command.type !== "prompt") {
    return command.description;
  }

  if (command.source === "plugin") {
    return command.pluginInfo?.repository ? `${command.description} (plugin:${command.pluginInfo.repository})` : `${command.description} (plugin)`;
  }

  if (command.source === "builtin" || command.source === "mcp") {
    return command.description;
  }

  if (command.source === "bundled") {
    return `${command.description} (bundled)`;
  }

  return `${command.description} (${getHumanReadableSource(command.source)})`; // Assuming Il is defined elsewhere
}


var allAvailableCommands;
var commandNameSet;
var getFilteredAndSortedCommands;
var getPromptSkillMetadata;
var getFilteredSlashCommandSkills;


// Initialization function for command definitions and related data.
var initializeCommandDefinitions = (() => {
  initializeConstants(); // fV1(), G2K()
  initializeCommandFlags(); // M2K(), n2K(), o2K()
  initializeEnvironmentVariables(); // FV1()
  initializeAutocompleteCommands(); // zzK(), wzK()
  initializeHelpCommands(); // HzK(), JzK(), OzK()
  initializeTerminalCommands(); // zwK(), _wK(), GwK()
  initializeVcsCommands(); // WwK(), EwK(), uwK(), pwK(), twK()
  initializeAuthCommands(); // HHK(), DHK(), MHK(), NHK()
  initializeProjectCommands(); //vG1(), yZ1(), KJK(), zJK()
  initializeRemoteCommands(); // A0K(), JOK(), XOK(), OOK(), _OK(), VOK(), NOK()
  initializeSettingsCommands(); // g$K() ,xB6()
  initializeHelpCommands(); // Q$K(), U$K(), l$K(), n$K(), O_K(), $_K(), G_K(), D_K()
  initializeAgentsCommands(); // I8A(), M_K(), V_K(), N_K(), lB6(), L_K(), g_K(), Q_K(), c_K()
  initializeUtilityCommands(); // bt(), a_K(), NZK(), vZK(), kZK()
  initializeIntegrations(); // vGK(), kGK(), LGK(), yGK(), IGK(), bGK(), xGK(), BGK(), lGK()
  initializePluginsCommands(); // XWK(), $WK()
  initializeFileCache(); // I1(), P1()
  initializeCompletionProviders() // ks(), pMA()
  initializeUsageReports(); // OQA()
  initializeReleaseNotes(); // Q7()
  initializePlatform(); // U4()
  initializeLsp(); // Lu6()
  initializeNetworkStatus(); // NWK(), vWK(), CWK(), RWK(), xWK()
  initializeNotifications(); // hu6(), LmA(), mWK(), FWK()
  initializeVimMode(); // QWK()
  initializeProduct(); // PDK()


  initializeUsageDataPaths();
  initializeProjectSetup();
  registerServices(); // eJ()

  allAvailableCommands = memoize(() => [
    Z2K, TGK, j2K, JWK, qzK, YzK, XzK, YwK, OwK, $wK, ZwK, BwK, kN1, TZK, EZK, wHK, WHK, jHK, fHK, AJK, YJK, eJK, swK, HOK, TWK, LWK, bWK, EGK, $OK, POK, fOK, m$K, F$K, c$K, MDK, i$K, gWK, OWK, kWK, P_K, Z_K, r2K, JN1, CGK, W_K, y8A, DpA, Op, uWK, j_K, f_K, E_K, C_K, m_K, F_K, o_K, fZK, fWK, cGK,
    ...(!isProductionBuild() ? [vx7, $Q7()] : []), // Assuming qB() is defined elsewhere
    X_K,
    ...[]
  ]);

  commandNameSet = memoize(() => new Set(allAvailableCommands().map(cmd => cmd.name)));

  getFilteredAndSortedCommands = memoize(async filter => {
    const [{
      skillDirCommands
    }] = await getSkillCommands(filter);
    return [...skillDirCommands];
  });
});