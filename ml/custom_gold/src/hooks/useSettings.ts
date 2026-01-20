import { getSettings, updateSettings as updateSettingsService } from '../services/terminal/settings.js';

export function useSettings() {
    return {
        getSettings,
        updateSettings: updateSettingsService
    };
}
