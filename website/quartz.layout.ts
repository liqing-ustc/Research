import { PageLayout, SharedLayout } from "./quartz/cfg"
import * as Component from "./quartz/components"

// components shared across all pages
export const sharedPageComponents: SharedLayout = {
  head: Component.Head(),
  header: [],
  afterBody: [],
  footer: Component.Footer({
    links: {
      GitHub: "https://github.com/liqing-ustc/MindFlow",
    },
  }),
}

// components for pages that display a single page (e.g. a single note)
export const defaultContentPageLayout: PageLayout = {
  beforeBody: [
    Component.Breadcrumbs(),
    Component.ArticleTitle(),
    Component.ContentMeta(),
    Component.TagList(),
  ],
  left: [
    Component.PageTitle(),
    Component.MobileOnly(Component.Spacer()),
    Component.Flex({
      components: [
        { Component: Component.Search(), grow: true },
        { Component: Component.Darkmode() },
      ],
    }),
    Component.Explorer({
      folderDefaultState: "collapsed",
      useSavedState: false,
      sortFn: (a, b) => {
        const order = ["Papers", "Topics", "Ideas", "Meetings", "Resources"]
        const aIdx = order.indexOf(a.name)
        const bIdx = order.indexOf(b.name)
        if (aIdx !== -1 && bIdx !== -1) return aIdx - bIdx
        if (aIdx !== -1) return -1
        if (bIdx !== -1) return 1
        return a.name.localeCompare(b.name)
      },
    }),
  ],
  right: [
    Component.DesktopOnly(Component.TableOfContents()),
    Component.Graph({
      localGraph: {
        depth: 1,
        showTags: false,
      },
    }),
  ],
  afterBody: [
    Component.Comments({
      provider: "giscus",
      options: {
        repo: "liqing-ustc/MindFlow",
        repoId: "R_kgDORucr2w",
        category: "Comments",
        categoryId: "DIC_kwDORucr284C5LSt",
        mapping: "pathname",
        reactionsEnabled: true,
        inputPosition: "bottom",
      },
    }),
  ],
}

// components for pages that display lists of pages  (e.g. tags or folders)
export const defaultListPageLayout: PageLayout = {
  beforeBody: [Component.Breadcrumbs(), Component.ArticleTitle(), Component.ContentMeta()],
  left: [],
  right: [],
}
