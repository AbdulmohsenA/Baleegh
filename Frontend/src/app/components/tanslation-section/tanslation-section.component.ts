import { FormsModule } from '@angular/forms';
import { ChangeDetectorRef, Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { EnglishBoxComponent } from './english-box/english-box.component';
import { ArabicBoxComponent } from './arabic-box/arabic-box.component';
import {
  debounceTime,
  distinctUntilChanged,
  of,
  Subject,
  switchMap,
} from 'rxjs';
import { BaleeghTranslateService } from '../../services/baleegh-translate.service';
import { ToastrService } from 'ngx-toastr';

@Component({
  selector: 'app-tanslation-section',
  standalone: true,
  imports: [CommonModule, FormsModule, EnglishBoxComponent, ArabicBoxComponent],
  templateUrl: './tanslation-section.component.html',
  styleUrl: './tanslation-section.component.scss',
})
export class TanslationSectionComponent {
  private textSubject = new Subject<string>();
  isLoading = false;
  translatedText: string = '';

  constructor(
    private baleeghTranslateService: BaleeghTranslateService,
    private toastr: ToastrService
  ) {
    this.textSubject
      .pipe(
        debounceTime(800),
        distinctUntilChanged(),
        switchMap((value) => {
          this.isLoading = true;
          return baleeghTranslateService.getBaleeghTranslatation(value);
        })
      )
      .subscribe({
        next: (response) => {
          this.translatedText = response.translation;
          this.isLoading = false;
        },
        error: (error) => {
          this.toastr.error('!Something went wrong', 'عذرًا، هناك مشكلة!');
          console.error('Translation error:', error);
          this.isLoading = false;
        },
      });
  }
  onTextChange(value: string): void {
    this.textSubject.next(value);
  }
}
