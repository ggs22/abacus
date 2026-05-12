const observer = new MutationObserver(() => {
    document.querySelectorAll('.dash-dropdown-content').forEach(el => {
        el.style.setProperty('max-height', '500px', 'important');
    });
});

observer.observe(document.body, { childList: true, subtree: true, attributes: true, attributeFilter: ['style'] });
