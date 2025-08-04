// allows the user to remove the flash message manually 
// automatically removes the flash message after 5 seconds
document.addEventListener("DOMContentLoaded", function () {
    document.querySelectorAll(".notification").forEach(function (notification) {
        // Auto-remove after 5 seconds (5000ms)
        setTimeout(() => {
            notification.style.opacity = "0"; // Fade out effect
            setTimeout(() => notification.remove(), 300); // Remove after fade-out
        }, 5000);

        // Allow manual dismissal
        let closeIcon = notification.querySelector(".close-notification");
        if (closeIcon) {
            closeIcon.addEventListener("click", function () {
                notification.style.opacity = "0";
                setTimeout(() => notification.remove(), 300);
            });
        }
    });
});