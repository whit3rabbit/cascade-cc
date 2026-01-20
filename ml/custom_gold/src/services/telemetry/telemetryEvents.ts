import { join } from "node:path";
import { getConfigDir } from "../../utils/settings/runtimeSettingsAndAuth.js";

/**
 * GitHub Actions Metadata Codec
 * Deobfuscated from csA in chunk_187.ts.
 */
export const githubActionsMetadataCodec = {
    fromJSON(json: any) {
        return {
            actor_id: json.actor_id ? String(json.actor_id) : "",
            repository_id: json.repository_id ? String(json.repository_id) : "",
            repository_owner_id: json.repository_owner_id ? String(json.repository_owner_id) : ""
        };
    },
    toJSON(data: any) {
        const json: any = {};
        if (data.actor_id !== undefined) json.actor_id = data.actor_id;
        if (data.repository_id !== undefined) json.repository_id = data.repository_id;
        if (data.repository_owner_id !== undefined) json.repository_owner_id = data.repository_owner_id;
        return json;
    }
};

/**
 * Environment Metadata Codec
 * Deobfuscated from lsA in chunk_187.ts.
 */
export const environmentMetadataCodec = {
    fromJSON(json: any) {
        return {
            platform: json.platform ? String(json.platform) : "",
            node_version: json.node_version ? String(json.node_version) : "",
            terminal: json.terminal ? String(json.terminal) : "",
            package_managers: json.package_managers ? String(json.package_managers) : "",
            runtimes: json.runtimes ? String(json.runtimes) : "",
            is_running_with_bun: Boolean(json.is_running_with_bun),
            is_ci: Boolean(json.is_ci),
            is_claubbit: Boolean(json.is_claubbit),
            is_github_action: Boolean(json.is_github_action),
            is_claude_code_action: Boolean(json.is_claude_code_action),
            is_claude_ai_auth: Boolean(json.is_claude_ai_auth),
            version: json.version ? String(json.version) : "",
            github_event_name: json.github_event_name ? String(json.github_event_name) : "",
            github_actions_runner_environment: json.github_actions_runner_environment ? String(json.github_actions_runner_environment) : "",
            github_actions_runner_os: json.github_actions_runner_os ? String(json.github_actions_runner_os) : "",
            github_action_ref: json.github_action_ref ? String(json.github_action_ref) : "",
            wsl_version: json.wsl_version ? String(json.wsl_version) : "",
            github_actions_metadata: json.github_actions_metadata ? githubActionsMetadataCodec.fromJSON(json.github_actions_metadata) : undefined,
            arch: json.arch ? String(json.arch) : "",
            is_claude_code_remote: Boolean(json.is_claude_code_remote),
            remote_environment_type: json.remote_environment_type ? String(json.remote_environment_type) : "",
            claude_code_container_id: json.claude_code_container_id ? String(json.claude_code_container_id) : "",
            claude_code_remote_session_id: json.claude_code_remote_session_id ? String(json.claude_code_remote_session_id) : "",
            tags: Array.isArray(json.tags) ? json.tags.map(String) : [],
            deployment_environment: json.deployment_environment ? String(json.deployment_environment) : ""
        };
    },
    toJSON(data: any) {
        const json: any = { ...data };
        if (data.github_actions_metadata) {
            json.github_actions_metadata = githubActionsMetadataCodec.toJSON(data.github_actions_metadata);
        }
        return json;
    }
};

/**
 * Core Telemetry Event Codec
 * Deobfuscated from isA in chunk_187.ts.
 */
export const telemetryEventCodec = {
    fromJSON(json: any) {
        return {
            event_name: json.event_name ? String(json.event_name) : "",
            client_timestamp: json.client_timestamp ? new Date(json.client_timestamp) : undefined,
            model: json.model ? String(json.model) : "",
            session_id: json.session_id ? String(json.session_id) : "",
            user_type: json.user_type ? String(json.user_type) : "",
            betas: json.betas ? String(json.betas) : "",
            env: json.env ? environmentMetadataCodec.fromJSON(json.env) : undefined,
            entrypoint: json.entrypoint ? String(json.entrypoint) : "",
            agent_sdk_version: json.agent_sdk_version ? String(json.agent_sdk_version) : "",
            is_interactive: Boolean(json.is_interactive),
            client_type: json.client_type ? String(json.client_type) : "",
            process: json.process ? String(json.process) : "",
            additional_metadata: json.additional_metadata ? String(json.additional_metadata) : "",
            server_timestamp: json.server_timestamp ? new Date(json.server_timestamp) : undefined,
            event_id: json.event_id ? String(json.event_id) : "",
            device_id: json.device_id ? String(json.device_id) : "",
            swe_bench_run_id: json.swe_bench_run_id ? String(json.swe_bench_run_id) : "",
            swe_bench_instance_id: json.swe_bench_instance_id ? String(json.swe_bench_instance_id) : "",
            swe_bench_task_id: json.swe_bench_task_id ? String(json.swe_bench_task_id) : "",
            email: json.email ? String(json.email) : ""
        };
    },
    toJSON(data: any) {
        const json: any = { ...data };
        if (data.client_timestamp) json.client_timestamp = data.client_timestamp.toISOString();
        if (data.server_timestamp) json.server_timestamp = data.server_timestamp.toISOString();
        if (data.env) json.env = environmentMetadataCodec.toJSON(data.env);
        return json;
    }
};

/**
 * Statsig Event Codec
 * Deobfuscated from xp1 in chunk_187.ts.
 */
export const statsigEventCodec = {
    fromJSON(json: any) {
        return {
            event_id: json.event_id ? String(json.event_id) : "",
            event_timestamp: json.event_timestamp ? new Date(json.event_timestamp) : undefined,
            timestamp: json.timestamp ? new Date(json.timestamp) : undefined,
            experiment_id: json.experiment_id ? String(json.experiment_id) : "",
            variation_id: json.variation_id ? Number(json.variation_id) : 0,
            environment: json.environment ? String(json.environment) : "",
            user_attributes: json.user_attributes ? String(json.user_attributes) : "",
            experiment_metadata: json.experiment_metadata ? String(json.experiment_metadata) : "",
            device_id: json.device_id ? String(json.device_id) : "",
            session_id: json.session_id ? String(json.session_id) : ""
        };
    },
    toJSON(data: any) {
        const json: any = { ...data };
        if (data.event_timestamp) json.event_timestamp = data.event_timestamp.toISOString();
        if (data.timestamp) json.timestamp = data.timestamp.toISOString();
        return json;
    }
};

/**
 * Returns the telemetry storage directory.
 * Deobfuscated from aZA in chunk_187.ts.
 */
export function getTelemetryDir(): string {
    return join(getConfigDir(), "telemetry");
}
