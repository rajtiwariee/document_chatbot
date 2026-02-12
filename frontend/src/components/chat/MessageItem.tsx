import { Avatar } from "../ui/Avatar";
import { cn } from "../../lib/utils";
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { User, Bot } from "lucide-react";

interface MessageItemProps {
  role: 'user' | 'assistant';
  content: string;
}

export function MessageItem({ role, content }: MessageItemProps) {
  const isUser = role === 'user';

  return (
    <div className={cn("flex w-full gap-4 p-4", isUser && "justify-end")}>
      {!isUser && (
        <Avatar className="h-8 w-8 border border-border bg-background">
           <div className="flex h-full w-full items-center justify-center bg-primary/10 text-primary">
             <Bot className="h-4 w-4" />
           </div>
        </Avatar>
      )}

      <div className={cn(
        "flex max-w-[80%] flex-col gap-2 rounded-lg px-4 py-3 text-sm",
        isUser 
          ? "bg-primary text-primary-foreground" 
          : "bg-muted/50 text-foreground"
      )}>
        <ReactMarkdown 
          remarkPlugins={[remarkGfm]}
          components={{
            // Style markdown elements to match the theme
            p: ({node, ...props}) => <p className="mb-2 last:mb-0" {...props} />,
            a: ({node, ...props}) => <a className="underline hover:text-primary" {...props} />,
            ul: ({node, ...props}) => <ul className="mb-2 list-disc pl-4" {...props} />,
            ol: ({node, ...props}) => <ol className="mb-2 list-decimal pl-4" {...props} />,
            code: ({node, className, children, ...props}) => {
                const match = /language-(\w+)/.exec(className || '')
                return match ? (
                  <pre className="rounded bg-black/50 p-2 overflow-x-auto text-xs my-2">
                    <code className={className} {...props}>
                      {children}
                    </code>
                  </pre>
                ) : (
                  <code className="rounded bg-black/20 px-1 py-0.5 text-xs" {...props}>
                    {children}
                  </code>
                )
            }
          }}
        >
          {content}
        </ReactMarkdown>
      </div>

      {isUser && (
        <Avatar className="h-8 w-8 bg-secondary">
          <div className="flex h-full w-full items-center justify-center">
            <User className="h-4 w-4" />
          </div>
        </Avatar>
      )}
    </div>
  );
}
