import Link from "next/link";

const links = [
  { href: "/", label: "Home" },
  { href: "/data-explorer", label: "Data Explorer" },
  { href: "/matchup", label: "Matchup Console" },
  { href: "/context", label: "Context Simulator" },
  { href: "/model-metrics", label: "Model Performance" },
  { href: "/glossary", label: "Glossary" },
];

export default function Nav() {
  return (
    <header className="nav">
      <div className="nav-logo">
        <Link href="/">B-ball Analysts</Link>
        <span>Basketball Strategy PSPI</span>
      </div>
      <nav className="nav-links">
        {links.map((link) => (
          <Link key={link.href} href={link.href}>
            {link.label}
          </Link>
        ))}
      </nav>
    </header>
  );
}
