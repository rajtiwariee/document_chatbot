import { clsx, type ClassValue } from "clsx"
import { twMerge } from "tailwind-merge"
import type { ConversationSummary } from "../types"

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

export function getRelativeTime(dateStr: string): string {
  const now = Date.now();
  const date = new Date(dateStr).getTime();
  const diffMs = now - date;
  const mins = Math.floor(diffMs / 60_000);
  const hours = Math.floor(diffMs / 3_600_000);
  const days = Math.floor(diffMs / 86_400_000);

  if (mins < 1) return 'Just now';
  if (mins < 60) return `${mins}m ago`;
  if (hours < 24) return `${hours}h ago`;

  const nowDate = new Date(); nowDate.setHours(0, 0, 0, 0);
  const itemDate = new Date(dateStr); itemDate.setHours(0, 0, 0, 0);
  const dayDiff = Math.round((nowDate.getTime() - itemDate.getTime()) / 86_400_000);

  if (dayDiff === 1) return 'Yesterday';
  if (days < 7) return `${days}d ago`;
  if (days < 30) return `${Math.floor(days / 7)}w ago`;
  return `${Math.floor(days / 30)}mo ago`;
}

export function groupConversationsByTime(
  conversations: ConversationSummary[]
): { label: string; items: ConversationSummary[] }[] {
  const buckets: { label: string; items: ConversationSummary[] }[] = [];
  const labelIndex: Record<string, number> = {};

  const now = new Date();
  const todayStart = new Date(now); todayStart.setHours(0, 0, 0, 0);
  const yesterdayStart = new Date(todayStart); yesterdayStart.setDate(yesterdayStart.getDate() - 1);

  for (const conv of conversations) {
    const d = new Date(conv.updated_at);
    let label: string;

    if (d >= todayStart) {
      label = 'Today';
    } else if (d >= yesterdayStart) {
      label = 'Yesterday';
    } else {
      const dayDiff = Math.floor((todayStart.getTime() - d.getTime()) / 86_400_000);
      if (dayDiff < 7) {
        label = 'Previous 7 days';
      } else if (dayDiff < 30) {
        label = 'Previous 30 days';
      } else {
        label = d.toLocaleDateString('en-US', { month: 'long', year: 'numeric' });
      }
    }

    if (labelIndex[label] === undefined) {
      labelIndex[label] = buckets.length;
      buckets.push({ label, items: [] });
    }
    buckets[labelIndex[label]].items.push(conv);
  }

  return buckets;
}
