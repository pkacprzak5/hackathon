import { VideoFeed } from "@/components/session/video-feed";
import { PARTICIPANT_COLORS } from "@/lib/mock-data";
import type { Participant } from "@/lib/types";
import { cn } from "@/lib/utils";

interface CameraGridProps {
  participants: Participant[];
  className?: string;
}

export function CameraGrid({ participants, className }: CameraGridProps) {
  const count = participants.length;

  if (count === 0) {
    return (
      <div className={cn("flex items-center justify-center rounded-xl bg-camera-bg p-8", className)}>
        <p className="text-sm text-text-secondary">Waiting for participants...</p>
      </div>
    );
  }

  if (count === 1) {
    return (
      <div className={cn("flex flex-col gap-2", className)}>
        <VideoFeed
          name={participants[0].username}
          score={participants[0].score}
          color={PARTICIPANT_COLORS[participants[0].id]}
          className="aspect-[4/3] w-full"
        />
      </div>
    );
  }

  if (count === 2) {
    return (
      <div className={cn("flex gap-2", className)}>
        {participants.map((p) => (
          <VideoFeed
            key={p.id}
            name={p.username}
            score={p.score}
            color={PARTICIPANT_COLORS[p.id]}
            className="aspect-[3/4] flex-1"
          />
        ))}
      </div>
    );
  }

  if (count === 3) {
    return (
      <div className={cn("flex flex-col gap-2", className)}>
        <div className="flex gap-2">
          {participants.slice(0, 2).map((p) => (
            <VideoFeed
              key={p.id}
              name={p.username}
              score={p.score}
              color={PARTICIPANT_COLORS[p.id]}
              className="aspect-[4/3] flex-1"
            />
          ))}
        </div>
        <VideoFeed
          name={participants[2].username}
          score={participants[2].score}
          color={PARTICIPANT_COLORS[participants[2].id]}
          className="aspect-[16/9] w-full"
        />
      </div>
    );
  }

  const shown = participants.slice(0, 4);
  return (
    <div className={cn("grid grid-cols-2 gap-2", className)}>
      {shown.map((p) => (
        <VideoFeed
          key={p.id}
          name={p.username}
          score={p.score}
          color={PARTICIPANT_COLORS[p.id]}
          className="aspect-[4/3]"
        />
      ))}
    </div>
  );
}
