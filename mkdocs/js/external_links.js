/**
 * Open external links in the site navigation (top tabs + sidebar) in a new
 * tab. "External" means an absolute http(s) href pointing at a different
 * origin than the docs site (e.g. the Google Colab "Try Pandera" notebook).
 *
 * Material for MkDocs does not set target=_blank on external nav links by
 * default, so this makes the expected behavior consistent with the up-right
 * arrow shown on those items.
 */
(function () {
  "use strict";

  function isExternal(href) {
    if (!href || !/^https?:\/\//i.test(href)) {
      return false;
    }
    try {
      return new URL(href, window.location.href).origin !== window.location.origin;
    } catch (err) {
      return true; // unparsable absolute URL → treat as external
    }
  }

  document.addEventListener("DOMContentLoaded", function () {
    document
      .querySelectorAll(".md-tabs a, .md-nav a")
      .forEach(function (link) {
        var href = link.getAttribute("href") || "";
        if (isExternal(href) && !link.getAttribute("target")) {
          link.setAttribute("target", "_blank");
          link.setAttribute("rel", "noopener");
        }
      });
  });
})();
