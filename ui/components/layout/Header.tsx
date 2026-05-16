import Link from "next/link";
import Image from "next/image";

export default function Header() {
  return (
    <header className="border-b border-slate-200 bg-white/80 backdrop-blur-sm sticky top-0 z-50">
      <div className="mx-auto flex max-w-3xl items-center px-6 py-3">
        <a href="/" className="flex items-center gap-2.5 group">
          <Image
            src="/logo.png"
            alt="InterviewPrep AI Logo"
            width={36}
            height={36}
            className="rounded-lg"
          />
          <span className="text-base font-semibold text-slate-900 group-hover:text-indigo-600 transition-colors">
            InterviewPrep AI
          </span>
        </a>
        <span className="ml-3 rounded-full bg-indigo-50 px-2.5 py-0.5 text-[11px] font-medium text-indigo-600 border border-indigo-100">
          Chat
        </span>
        <div className="ml-auto">
          <Link
            href="/about"
            className="text-sm font-medium text-slate-600 hover:text-indigo-600 transition-colors"
          >
            About
          </Link>
        </div>
      </div>
    </header>
  );
}
