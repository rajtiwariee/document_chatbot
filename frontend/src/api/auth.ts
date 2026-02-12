import { client } from './client';
import type { AuthResponse, User } from '../types';

export const auth = {
  login: async (formData: FormData): Promise<AuthResponse> => {
    // fastAPI OAuth2PasswordRequestForm expects form data
    const response = await client.post<AuthResponse>('/auth/login', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });
    return response.data;
  },

  register: async (data: any): Promise<User> => {
    const response = await client.post<User>('/auth/register', data);
    return response.data;
  },

  getMe: async (): Promise<User> => {
    const response = await client.get<User>('/auth/me');
    return response.data;
  },
};
