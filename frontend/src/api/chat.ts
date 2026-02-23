import { client } from './client';
import type { ConversationSummary, ChatMessage } from '../types';


export interface ChatRequest {
  message: string;
  conversation_id?: string;
  files?: File[];
}

export interface ChatResponse {
  message: string;
  conversation_id: string;
  sources?: any[];
}

export interface ConversationsResponse {
  conversations: ConversationSummary[];
}

export interface ApiAttachment {
  id: string;
  filename: string;
  mime_type: string;
  data_url?: string;  // local: inline base64 data URL
  url?: string;       // GCS: /api/chat/attachments?...
}

export interface ConversationDetail {
  id: string;
  title: string;
  messages: Array<{
    id: string;
    role: 'user' | 'assistant';
    content: string;
    sources?: any[];
    created_at?: string;
    attachments?: ApiAttachment[];
  }>;
}

export const chat = {
  sendMessage: async (data: ChatRequest): Promise<ChatResponse> => {
    if (data.files && data.files.length > 0) {
      const formData = new FormData();
      formData.append('message', data.message);
      if (data.conversation_id) {
        formData.append('conversation_id', data.conversation_id);
      }
      for (const file of data.files) {
        formData.append('files', file);
      }
      const response = await client.post<ChatResponse>('/chat', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      return response.data;
    }

    const response = await client.post<ChatResponse>('/chat', {
      message: data.message,
      conversation_id: data.conversation_id,
    });
    return response.data;
  },

  getConversations: async (): Promise<ConversationSummary[]> => {
    const response = await client.get<ConversationsResponse>('/chat/conversations');
    return response.data.conversations;
  },

  getConversation: async (id: string): Promise<ConversationDetail> => {
    const response = await client.get<ConversationDetail>(`/chat/conversations/${id}`);
    return response.data;
  },

  getConversationMessages: async (id: string): Promise<ChatMessage[]> => {
    const data = await chat.getConversation(id);
    return data.messages.map(msg => ({
      id: msg.id,
      role: msg.role,
      content: msg.content,
      sources: msg.sources,
      created_at: msg.created_at,
      attachments: msg.attachments?.map(att => ({
        id: att.id,
        filename: att.filename,
        type: att.mime_type.startsWith('image/') ? 'image' as const
              : att.mime_type.includes('spreadsheetml') || att.mime_type === 'text/csv'
                ? 'spreadsheet' as const
                : 'document' as const,
        preview: att.data_url ?? att.url,
        useAuthFetch: !!att.url && !att.data_url,
      })) ?? [],
    }));
  },

  deleteConversation: async (id: string): Promise<void> => {
    await client.delete(`/chat/conversations/${id}`);
  }
};
