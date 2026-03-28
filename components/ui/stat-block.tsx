import { cn } from "@/lib/utils";

interface StatBlockProps {
  label: string;
  value: string;
  valueClassName?: string;
  className?: string;
}

export function StatBlock({ label, value, valueClassName, className }: StatBlockProps) {
  return (
    <div className={cn("flex flex-col", className)}>
      <span className={cn("text-sm font-bold text-text-primary", valueClassName)}>{value}</span>
      <span className="text-[10px] text-text-secondary">{label}</span>
    </div>
  );
}
