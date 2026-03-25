import { pathToRoot } from "../util/path"
import { QuartzComponent, QuartzComponentConstructor, QuartzComponentProps } from "./types"

const TopNav: QuartzComponent = ({ fileData }: QuartzComponentProps) => {
  const root = pathToRoot(fileData.slug!)
  const sections = [
    { label: "Papers", path: "Papers/" },
    { label: "Ideas", path: "Ideas/" },
    { label: "Topics", path: "Topics/" },
    { label: "Projects", path: "Projects/" },
    { label: "Meetings", path: "Meetings/" },
  ]
  return (
    <nav class="top-nav">
      {sections.map(({ label, path }) => (
        <a href={`${root}${path}`}>{label}</a>
      ))}
    </nav>
  )
}

TopNav.css = `
/* Header bar layout: PageTitle | TopNav (flex:1) | Search | Darkmode */
#quartz-header {
  display: flex;
  align-items: center;
  gap: 1rem;
  padding: 0.6rem 0;
  border-bottom: 1px solid var(--lightgray);
  margin-bottom: 0;
}

#quartz-header .page-title {
  flex-shrink: 0;
  font-size: 1.2rem;
  margin-right: 0.5rem;
}

.top-nav {
  display: flex;
  gap: 1.25rem;
  flex: 1;
}

.top-nav a {
  color: var(--darkgray);
  text-decoration: none;
  font-size: 0.85rem;
  font-weight: 600;
  letter-spacing: 0.02em;
  opacity: 0.75;
  transition: color 0.15s ease, opacity 0.15s ease;
  white-space: nowrap;
}

.top-nav a:hover {
  color: var(--secondary);
  opacity: 1;
}

/* Push search to the right */
#quartz-header > .search {
  margin-left: auto;
}
`

export default (() => TopNav) satisfies QuartzComponentConstructor
