import { CommonModule } from '@angular/common';
import { Component, HostListener } from '@angular/core';
import { RouterModule } from '@angular/router';
import { CookieService } from 'ngx-cookie-service';

@Component({
  selector: 'app-navbar',
  standalone: true,
  imports: [CommonModule,RouterModule],
  templateUrl: './navbar.component.html',
  styleUrl: './navbar.component.scss'
})
export class NavbarComponent {
  navbarScrolled: boolean = false;
  isDarkTheme = false; // Default is light theme

  constructor(private cookieService: CookieService) {}
  
  ngOnInit() {
    const darkThemeCookie = this.cookieService.get('darkTheme');
    this.isDarkTheme = darkThemeCookie === 'true';
    this.applyTheme();
  }
  
  toggleTheme() {
    this.isDarkTheme = !this.isDarkTheme;
    this.applyTheme();
    this.cookieService.set('darkTheme', String(this.isDarkTheme), { expires: 9999 });
  }
  
  applyTheme() {
    if (this.isDarkTheme) {
      document.body.classList.add('dark-theme');
    } else {
      document.body.classList.remove('dark-theme');
    }
  }
  @HostListener('window:scroll', [])
  onWindowScroll() {
    const scrollOffset =
      window.pageYOffset ||
      document.documentElement.scrollTop ||
      document.body.scrollTop ||
      0;
    this.navbarScrolled = scrollOffset > 0;
  }

}
