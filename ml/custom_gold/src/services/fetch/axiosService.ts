
import axios, { AxiosInstance, AxiosRequestConfig } from 'axios';

export function createAxiosInstance(config?: AxiosRequestConfig): AxiosInstance {
    return axios.create({
        timeout: 30000,
        ...config,
        headers: {
            'User-Agent': 'ClaudeCode/1.0',
            ...config?.headers
        }
    });
}

export const SQ = createAxiosInstance();
export const axiosInstance = SQ;
