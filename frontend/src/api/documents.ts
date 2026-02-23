import { client } from './client';
import type { Document } from '../types';

export interface DocumentsResponse {
  documents: Document[];
}

export interface UploadResponse {
  id: string;
  filename: string;
  status: string;
  message: string;
}

export const documents = {
  getAll: async (): Promise<Document[]> => {
    const response = await client.get<DocumentsResponse>('/documents');
    return response.data.documents;
  },

  upload: async (file: File): Promise<UploadResponse> => {
    const formData = new FormData();
    formData.append('file', file);
    
    const response = await client.post<UploadResponse>('/documents/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  },

  download: async (id: string, inline = false): Promise<{ blob: Blob; filename: string }> => {
    const response = await client.get(
      `/documents/${id}/download${inline ? '?inline=true' : ''}`,
      { responseType: 'blob' },
    );
    const cd = (response.headers['content-disposition'] as string) || '';
    const match = cd.match(/filename="([^"]+)"/);
    return { blob: response.data, filename: match?.[1] ?? 'document' };
  },

  delete: async (id: string): Promise<void> => {
    await client.delete(`/documents/${id}`);
  }
};
