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

export interface ConversationDetail {
  id: string;
  title: string;
  messages: ChatMessage[];
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

  deleteConversation: async (id: string): Promise<void> => {
    await client.delete(`/chat/conversations/${id}`);
  }
};
