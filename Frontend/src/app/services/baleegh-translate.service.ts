import { Injectable } from '@angular/core';
import { environment } from '../../environments/environment';
import { catchError, Observable, Subject, throwError } from 'rxjs';
import { HttpClient } from '@angular/common/http';

@Injectable({
  providedIn: 'root',
})
export class BaleeghTranslateService {
  constructor(private http: HttpClient) {}
  getBaleeghTranslatation(text: string): Observable<{ translation: string }> {
    if (text) {
      return this.http.get<any>(environment.apiUrl + `?text=${text}`).pipe(
        catchError((error) => {
          console.error('An error occurred:', error.message);
          return throwError(() => new Error('Failed to translate.'));
        })
      );
    } else {
      return new Observable((observer) => {
        observer.next({ translation: '' });
      });
    }
  }
}
