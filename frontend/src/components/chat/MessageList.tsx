import { useRef, useEffect } from "react";
import { MessageItem } from "./MessageItem";
import type { ChatMessage } from "../../types";

interface MessageListProps {
  messages: ChatMessage[];
  isLoading?: boolean;
}

export function MessageList({ messages, isLoading }: MessageListProps) {
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isLoading]);

  return (
    <div className="flex-1 overflow-y-auto p-4 sm:p-6">
      <div className="mx-auto flex max-w-3xl flex-col gap-6">
        {messages.map((msg, index) => (
          <MessageItem 
            key={index} 
            role={msg.role} 
            content={msg.content} 
          />
        ))}
        {isLoading && (
          <div className="flex w-full gap-4 p-4">
             <div className="h-8 w-8 rounded-full bg-muted/50 animate-pulse" />
             <div className="h-12 w-32 rounded-lg bg-muted/50 animate-pulse" />
          </div>
        )}
        <div ref={bottomRef} />
      </div>
    </div>
  );
}
