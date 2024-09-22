// Add event listener for DOMContentLoaded event
window.addEventListener("DOMContentLoaded", function() {
    // Select all <a> elements with class "external"
    var externalLinks = document.querySelectorAll("a.external");

    // Loop through each <a> element with class "external"
    externalLinks.forEach(function(link) {
        // Set the target attribute to "_blank"
        link.setAttribute("target", "_blank");
    });
});


function setHtmlDataTheme() {
    // Set theme at the root html element
    setTimeout(() => {
      const theme = document.body.dataset.theme;
      const prefersDark = window.matchMedia("(prefers-color-scheme: dark)").matches;

      if (theme === "auto") {
        document.documentElement.dataset.theme = prefersDark ? "dark" : "light";
      } else {
        document.documentElement.dataset.theme = theme;
      }
    }, 10)
  }

function setupAlgoliaTheme() {
    // To get darkmode in the algolia search modal, we need to set the theme in
    // the root html element. This function propagates the theme set by furo
    // that's set in the body element.
    const buttons = document.getElementsByClassName("theme-toggle");

    // set for initial document load
    setHtmlDataTheme();

    // listen for when theme button is clicked.
    Array.from(buttons).forEach((btn) => {
      btn.addEventListener("click", setHtmlDataTheme);
    });
}

function main() {
    setupAlgoliaTheme()
}

document.addEventListener('DOMContentLoaded', main);
window.addEventListener('keydown', (event) => {
    if (event.code === "Escape") {
        // make sure to prevent default behavior with escape key so that algolia
        // modal can be closed properly.
        event.preventDefault();
    }
});
