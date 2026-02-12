import * as React from "react"
import { cn } from "../../lib/utils"

export interface AvatarProps extends React.HTMLAttributes<HTMLDivElement> {
  src?: string;
  fallback?: string;
}

export const Avatar = React.forwardRef<HTMLDivElement, AvatarProps>(
  ({ className, src, fallback, ...props }, ref) => {
    return (
      <div
        ref={ref}
        className={cn(
          "relative flex h-10 w-10 shrink-0 overflow-hidden rounded-full",
          className
        )}
        {...props}
      >
        {src ? (
          <img
            src={src}
            alt="Avatar"
            className="aspect-square h-full w-full object-cover"
          />
        ) : props.children ? (
          props.children
        ) : (
          <div className="flex h-full w-full items-center justify-center rounded-full bg-muted">
            <span className="text-sm font-medium uppercase text-muted-foreground">
              {fallback?.slice(0, 2) || "??"}
            </span>
          </div>
        )}
      </div>
    )
  }
)
Avatar.displayName = "Avatar"
