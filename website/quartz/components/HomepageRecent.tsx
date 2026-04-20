import { QuartzComponent, QuartzComponentConstructor, QuartzComponentProps } from "./types"
import { pathToRoot, resolveRelative } from "../util/path"
import { formatDate } from "./Date"

interface Options {
  limit: number
}

const defaultOptions: Options = {
  limit: 10,
}

export default ((userOpts?: Partial<Options>) => {
  const opts = { ...defaultOptions, ...userOpts }

  const HomepageRecent: QuartzComponent = ({ allFiles, fileData, cfg }: QuartzComponentProps) => {
    if (fileData.slug !== "index") return <></>

    const urlPath = cfg.baseUrl?.includes("/")
      ? "/" + cfg.baseUrl.split("/").slice(1).join("/")
      : pathToRoot(fileData.slug!)
    const base = urlPath.endsWith("/") ? urlPath : urlPath + "/"

    const pages = allFiles
      .filter(
        (f) =>
          f.slug &&
          f.slug !== "index" &&
          !f.slug.endsWith("/index") &&
          !f.slug.endsWith("/_index") &&
          f.dates?.modified,
      )
      .sort((a, b) => b.dates!.modified.getTime() - a.dates!.modified.getTime())
      .slice(0, opts.limit)

    if (pages.length === 0) return <></>

    return (
      <div class="homepage-recent">
        <h3>Recent</h3>
        <ul class="recent-list">
          {pages.map((page) => {
            const title = page.frontmatter?.title ?? page.slug
            const folder = page.slug!.includes("/") ? page.slug!.split("/")[0] : ""
            return (
              <li class="recent-item">
                <a href={resolveRelative(fileData.slug!, page.slug!)} class="recent-title internal">
                  {title}
                </a>
                <span class="recent-meta">
                  {folder && (
                    <a href={`${base}${folder}/`} class="recent-folder internal">
                      {folder}
                    </a>
                  )}
                  <time datetime={page.dates!.modified.toISOString()}>
                    {formatDate(page.dates!.modified, cfg.locale)}
                  </time>
                </span>
              </li>
            )
          })}
        </ul>
      </div>
    )
  }

  HomepageRecent.css = `
.homepage-recent {
  margin: 2rem 0;
}

.homepage-recent h3 {
  font-size: 1rem;
  font-weight: 600;
  color: var(--darkgray);
  margin: 0 0 0.75rem 0;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  opacity: 0.6;
}

.recent-list {
  list-style: none;
  padding: 0;
  margin: 0;
}

.recent-item {
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  gap: 1rem;
  padding: 0.4rem 0;
  border-bottom: 1px solid var(--lightgray);
}

.recent-item:last-child {
  border-bottom: none;
}

.recent-title {
  flex: 1;
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.recent-meta {
  display: inline-flex;
  align-items: baseline;
  gap: 0.6rem;
  font-size: 0.85rem;
  color: var(--gray);
  flex-shrink: 0;
}

.recent-folder {
  color: var(--secondary) !important;
  opacity: 0.8;
  text-decoration: none;
}

.recent-folder:hover {
  opacity: 1;
}
`

  return HomepageRecent
}) satisfies QuartzComponentConstructor
