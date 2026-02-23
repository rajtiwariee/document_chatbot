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

export interface ChatAttachment {
  id: string;
  file?: File;          // present for new uploads, absent for history
  preview?: string;     // blob URL (new) | data URL (local history) | API URL (GCS history)
  type: 'image' | 'document' | 'spreadsheet';
  filename?: string;    // used when file is absent (history case)
  useAuthFetch?: boolean; // true when preview is an API URL (GCS case, needs auth header)
}

export interface ChatMessage {
  id?: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp?: number;
  created_at?: string;
  sources?: any[];
  attachments?: ChatAttachment[];
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
