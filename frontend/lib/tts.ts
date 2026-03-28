export function speakCoaching(text: string): void {
  if (typeof window === "undefined") return;
  if (!("speechSynthesis" in window)) return;

  speechSynthesis.cancel();
  const utterance = new SpeechSynthesisUtterance(text);
  utterance.rate = 1.1;
  utterance.pitch = 1.0;
  speechSynthesis.speak(utterance);
}
