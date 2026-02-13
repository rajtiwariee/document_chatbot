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
        <Avatar className="h-9 w-9 border border-border bg-background shrink-0 mt-1 shadow-sm">
           <div className="flex h-full w-full items-center justify-center bg-primary/5 text-primary">
             <Bot className="h-5 w-5" />
           </div>
        </Avatar>
      )}

      <div className={cn(
        "flex max-w-[85%] flex-col gap-2 rounded-2xl px-5 py-3.5 text-sm shadow-sm leading-relaxed",
        isUser 
          ? "bg-primary text-primary-foreground rounded-br-sm" 
          : "bg-card border border-border text-card-foreground rounded-bl-sm"
      )}>
        <ReactMarkdown 
          remarkPlugins={[remarkGfm]}
          components={{
            // Style markdown elements to match the theme
            p: ({node, ...props}) => <p className="mb-2 last:mb-0" {...props} />,
            a: ({node, ...props}) => <a className="underline hover:text-primary/80 font-medium" {...props} />,
            ul: ({node, ...props}) => <ul className="mb-3 list-disc pl-4 space-y-1" {...props} />,
            ol: ({node, ...props}) => <ol className="mb-3 list-decimal pl-4 space-y-1" {...props} />,
            h1: ({node, ...props}) => <h1 className="text-lg font-bold my-2" {...props} />,
            h2: ({node, ...props}) => <h2 className="text-base font-semibold my-2" {...props} />,
            h3: ({node, ...props}) => <h3 className="text-sm font-semibold my-1" {...props} />,
            blockquote: ({node, ...props}) => <blockquote className="border-l-4 border-primary/20 pl-4 py-1 italic my-2 text-muted-foreground" {...props} />,
            code: ({node, className, children, ...props}) => {
                const match = /language-(\w+)/.exec(className || '')
                return match ? (
                  <pre className="rounded-lg bg-zinc-950 p-3 overflow-x-auto text-xs my-3 text-zinc-50 border border-zinc-800">
                    <code className={className} {...props}>
                      {children}
                    </code>
                  </pre>
                ) : (
                  <code className="rounded-md bg-muted px-1.5 py-0.5 text-xs font-mono border border-border/50" {...props}>
                    {children}
                  </code>
                )
            },
            table: ({node, ...props}) => <div className="overflow-x-auto my-3 rounded-lg border border-border"><table className="w-full text-sm" {...props} /></div>,
            th: ({node, ...props}) => <th className="bg-muted/50 px-3 py-2 text-left font-semibold border-b border-border" {...props} />,
            td: ({node, ...props}) => <td className="px-3 py-2 border-b border-border last:border-0" {...props} />,
          }}
        >
          {content}
        </ReactMarkdown>
      </div>

      {isUser && (
        <Avatar className="h-9 w-9 bg-primary/10 shrink-0 mt-1 shadow-sm">
          <div className="flex h-full w-full items-center justify-center text-primary">
            <User className="h-5 w-5" />
          </div>
        </Avatar>
      )}
    </div>
  );
}
