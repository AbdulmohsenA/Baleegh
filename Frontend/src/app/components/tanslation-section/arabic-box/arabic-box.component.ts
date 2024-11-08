import { Component, Input, input, OnChanges } from '@angular/core';
import { Clipboard } from '@angular/cdk/clipboard';
import { NgxSpinnerService, NgxSpinnerModule } from 'ngx-spinner';
import { ToastrModule, ToastrService } from 'ngx-toastr';

@Component({
  selector: 'app-arabic-box',
  standalone: true,
  imports: [NgxSpinnerModule],
  templateUrl: './arabic-box.component.html',
  styleUrls: [
    './arabic-box.component.scss',
    '../tanslation-section.component.scss',
  ],
})
export class ArabicBoxComponent implements OnChanges {
  @Input() translatedText: string = 'النص المترجم';
  @Input() isLoading: boolean = false;

  placeholderText: string = 'يرجى إدخال نص للترجمة';

  constructor(
    private clipboard: Clipboard,
    private spinner: NgxSpinnerService,
    private toastr: ToastrService
  ) {}

  copyText() {
    this.toastr.success('!Copied', 'تم النسخ!');
    this.clipboard.copy(this.translatedText);
  }
  ngOnChanges() {
    if (this.isLoading) {
      this.spinner.show();
    } else {
      this.spinner.hide();
    }
  }
}
