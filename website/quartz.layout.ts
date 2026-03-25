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
        {
          Component: Component.Search(),
          grow: true,
        },
        { Component: Component.Darkmode() },
      ],
    }),
    Component.Explorer({
      folderDefaultState: "collapsed",
      useSavedState: false,
    }),
    Component.DesktopOnly(Component.Graph({
      localGraph: {
        depth: 1,
        showTags: false,
      },
    })),
  ],
  right: [
    Component.DesktopOnly(Component.TableOfContents()),
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
  left: [
    Component.PageTitle(),
    Component.MobileOnly(Component.Spacer()),
    Component.Flex({
      components: [
        {
          Component: Component.Search(),
          grow: true,
        },
        { Component: Component.Darkmode() },
      ],
    }),
    Component.Explorer(),
  ],
  right: [],
}
