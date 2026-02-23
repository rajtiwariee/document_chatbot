import { useState, useEffect } from "react";
import { MessageList } from "./MessageList";
import { ChatInput } from "./ChatInput";
import { WelcomeScreen } from "./WelcomeScreen";
import type { ChatMessage, ChatAttachment } from "../../types";
import { chat } from "../../api/chat";

interface ChatInterfaceProps {
  conversationId: string | null;
  onConversationCreated: (id: string) => void;
}

export function ChatInterface({ conversationId, onConversationCreated }: ChatInterfaceProps) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isLoading, setIsLoading] = useState(false);

  // Load conversation when ID changes
  useEffect(() => {
    if (conversationId) {
      loadConversation(conversationId);
    } else {
      setMessages([]);
    }
  }, [conversationId]);

  const loadConversation = async (id: string) => {
    setIsLoading(true);
    try {
      const messages = await chat.getConversationMessages(id);
      setMessages(messages);
    } catch (error) {
      console.error("Failed to load conversation", error);
    } finally {
      setIsLoading(false);
    }
  };

  const handleSend = async (content: string, files?: File[]) => {
    // Build attachments for local display
    const attachments: ChatAttachment[] | undefined = files?.map(f => ({
      id: crypto.randomUUID(),
      file: f,
      preview: f.type.startsWith('image/') ? URL.createObjectURL(f) : undefined,
      type: f.type.startsWith('image/')
        ? 'image' as const
        : f.name.match(/\.(csv|xlsx)$/i)
          ? 'spreadsheet' as const
          : 'document' as const,
    }));

    // Add user message immediately
    const userMsg: ChatMessage = {
      role: 'user',
      content,
      timestamp: Date.now(),
      attachments,
    };
    setMessages(prev => [...prev, userMsg]);
    setIsLoading(true);

    try {
      const response = await chat.sendMessage({
        message: content,
        conversation_id: conversationId || undefined,
        files,
      });

      const aiMsg: ChatMessage = {
        role: 'assistant',
        content: response.message,
        timestamp: Date.now(),
        sources: response.sources
      };
      setMessages(prev => [...prev, aiMsg]);

      // If this was a new conversation, notify parent
      if (!conversationId && response.conversation_id) {
        onConversationCreated(response.conversation_id);
      }
    } catch (error) {
      console.error(error);
      const errorMsg: ChatMessage = {
        role: 'assistant',
        content: "Sorry, I encountered an error connecting to the server.",
        timestamp: Date.now()
      };
      setMessages(prev => [...prev, errorMsg]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="flex flex-1 flex-col h-full">
      {messages.length === 0 && !conversationId ? (
        <WelcomeScreen />
      ) : (
        <MessageList messages={messages} isLoading={isLoading} />
      )}
      <ChatInput onSend={handleSend} disabled={isLoading} />
    </div>
  );
}
