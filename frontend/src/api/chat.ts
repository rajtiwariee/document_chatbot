import { client } from './client';


export interface ChatRequest {
  message: string;
  conversation_id?: string;
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

import type { ConversationSummary, ChatMessage } from '../types';

export const chat = {
  sendMessage: async (data: ChatRequest): Promise<ChatResponse> => {
    const response = await client.post<ChatResponse>('/chat', data);
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
