import { MarketplaceService as MarketplaceServiceImpl } from '../marketplace/MarketplaceService.js';

export class MarketplaceService {
    static async checkAndInstallOfficialPlugins(): Promise<void> {
        await MarketplaceServiceImpl.autoInstallOfficialMarketplace();
    }
}
