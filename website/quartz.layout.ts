import { PageLayout, SharedLayout } from "./quartz/cfg"
import * as Component from "./quartz/components"

// components shared across all pages
export const sharedPageComponents: SharedLayout = {
  head: Component.Head(),
  header: [],
  afterBody: [
    Component.HomepageStats(),
    Component.HomepageRecent(),
    Component.HomepageTags(),
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
    Component.PaperMeta(),
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
      mapFn: (node) => {
        if (!node.isFolder) {
          node.displayName = node.slugSegment
        }
      },
      sortFn: (a, b) => {
        const order = ["Papers", "Topics", "Ideas", "Meetings", "Resources"]
        const aIdx = order.indexOf(a.displayName)
        const bIdx = order.indexOf(b.displayName)
        if (aIdx !== -1 && bIdx !== -1) return aIdx - bIdx
        if (aIdx !== -1) return -1
        if (bIdx !== -1) return 1
        return a.displayName.localeCompare(b.displayName)
      },
    }),
    Component.SidebarTags(),
  ],
  right: [
    Component.DesktopOnly(Component.TableOfContents()),
  ],
}

// components for pages that display lists of pages  (e.g. tags or folders)
export const defaultListPageLayout: PageLayout = {
  beforeBody: [Component.Breadcrumbs(), Component.ArticleTitle(), Component.ContentMeta()],
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
      mapFn: (node) => {
        if (!node.isFolder) {
          node.displayName = node.slugSegment
        }
      },
      sortFn: (a, b) => {
        const order = ["Papers", "Topics", "Ideas", "Meetings", "Resources"]
        const aIdx = order.indexOf(a.displayName)
        const bIdx = order.indexOf(b.displayName)
        if (aIdx !== -1 && bIdx !== -1) return aIdx - bIdx
        if (aIdx !== -1) return -1
        if (bIdx !== -1) return 1
        return a.displayName.localeCompare(b.displayName)
      },
    }),
    Component.SidebarTags(),
  ],
  right: [],
}
