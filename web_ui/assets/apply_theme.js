// Apply stored theme immediately before React renders to prevent flash.
(function () {
  var theme = localStorage.getItem("abacus-theme") || "light";
  document.documentElement.dataset.theme = theme;
})();
