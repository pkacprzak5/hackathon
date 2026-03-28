"use client";

import { ChevronRight, HelpCircle, LogOut, Shield, Trophy } from "lucide-react";
import { useRouter } from "next/navigation";

import { ThemeToggle } from "@/components/ui/theme-toggle";
import { useSession } from "@/hooks/use-session";
import { useSessionContext } from "@/providers/session-provider";

export default function ProfilePage() {
  const { username } = useSession();
  const { dispatch } = useSessionContext();
  const router = useRouter();

  function handleLogout() {
    dispatch({ type: "RESET" });
    router.replace("/");
  }

  const menuItems = [
    { icon: Trophy, label: "Leaderboard", href: "/leaderboard" },
    { icon: Shield, label: "Privacy", href: "#" },
    { icon: HelpCircle, label: "Help & Support", href: "#" },
  ];

  return (
    <div className="flex flex-col bg-bg-surface">
      <div className="px-5 pt-12 pb-4">
        <h1 className="text-2xl font-bold text-text-primary">Profile</h1>
      </div>

      {/* User card */}
      <div className="px-5">
        <div className="flex items-center gap-4 rounded-3xl bg-bg-card p-5 ring-1 ring-border-light">
          <div className="flex h-14 w-14 items-center justify-center rounded-2xl bg-linear-to-br from-gradient-start to-gradient-end text-xl font-bold text-white">
            {username[0]?.toUpperCase()}
          </div>
          <div>
            <p className="text-lg font-bold text-text-primary">{username}</p>
            <p className="text-sm text-text-secondary">Player</p>
          </div>
        </div>
      </div>

      {/* Theme toggle */}
      <div className="px-5 pt-4">
        <ThemeToggle />
      </div>

      {/* Menu items */}
      <div className="flex flex-col gap-2 px-5 pt-4">
        {menuItems.map((item) => (
          <button
            key={item.label}
            onClick={() => item.href !== "#" && router.push(item.href)}
            className="flex items-center gap-3 rounded-2xl bg-bg-card p-4 ring-1 ring-border-light"
          >
            <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-bg-surface">
              <item.icon className="h-5 w-5 text-text-secondary" />
            </div>
            <span className="flex-1 text-left text-sm font-medium text-text-primary">{item.label}</span>
            <ChevronRight className="h-4 w-4 text-text-muted" />
          </button>
        ))}
      </div>

      {/* Logout */}
      <div className="px-5 pt-4 pb-8">
        <button
          onClick={handleLogout}
          className="flex w-full items-center gap-3 rounded-2xl bg-error/10 p-4"
        >
          <LogOut className="h-5 w-5 text-error" />
          <span className="text-sm font-medium text-error">Log Out</span>
        </button>
      </div>
    </div>
  );
}
