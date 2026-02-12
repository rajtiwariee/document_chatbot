export interface User {
  id: string;
  email: string;
  is_active: boolean;
  tenant_id: string;
  created_at: string;
}

export interface AuthResponse {
  access_token: string;
  token_type: string;
  user: User;
}

export interface ChatMessage {
  id?: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp?: number;
  created_at?: string;
  sources?: any[];
}

export interface ConversationSummary {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
  message_count: number;
}

export interface Document {
  id: string;
  filename: string;
  status: 'pending' | 'processing' | 'completed' | 'failed';
  file_size: number;
  document_type: string;
  created_at: string;
  error_message?: string;
}
