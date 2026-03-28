"use client";

import { Moon, Sun } from "lucide-react";
import { useTheme } from "next-themes";
import { useSyncExternalStore } from "react";

import { cn } from "@/lib/utils";

const subscribe = () => () => {};
const getSnapshot = () => true;
const getServerSnapshot = () => false;

export function ThemeToggle({ className }: { className?: string }) {
  const { theme, setTheme } = useTheme();
  const mounted = useSyncExternalStore(subscribe, getSnapshot, getServerSnapshot);
  if (!mounted) return null;

  const isDark = theme === "dark";

  return (
    <button
      onClick={() => setTheme(isDark ? "light" : "dark")}
      className={cn(
        "flex items-center gap-3 rounded-2xl bg-bg-card p-4 w-full",
        className
      )}
    >
      <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-bg-surface">
        {isDark ? <Moon className="h-5 w-5 text-gradient-start" /> : <Sun className="h-5 w-5 text-warning" />}
      </div>
      <div className="flex-1 text-left">
        <p className="text-sm font-semibold text-text-primary">Dark Mode</p>
        <p className="text-xs text-text-secondary">{isDark ? "On" : "Off"}</p>
      </div>
      <div className={cn(
        "h-6 w-11 rounded-full p-0.5 transition-colors",
        isDark ? "bg-gradient-start" : "bg-border"
      )}>
        <div className={cn(
          "h-5 w-5 rounded-full bg-white transition-transform",
          isDark ? "translate-x-5" : "translate-x-0"
        )} />
      </div>
    </button>
  );
}
