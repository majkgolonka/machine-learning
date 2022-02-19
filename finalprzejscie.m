% clc;
% clear all;
% close all;
% %próba 1
% clearvars
% T = readtable('titanic.csv');
% TF = ismissing(T); %funkcja zwraca 1, jezeli znajdzie pusta komorke (brakujaca dana)
% %sprawdzamy, czy w ktorej kolumnie mamy 1 (czyli brakujace dane) - jesli tak, to suma wyjdzie wieksza niz 0
% sum(TF,1)
% T=T(:,[1:4,6:end])
% for i=1:120
% random_T = T(randperm(size(T, 1)), :); %zmieniamy kolejnosc obiektow na losowa, zeby uniknac duzych skupien jednej klasy
% training_T = random_T(1:floor(0.7*size(random_T,1)),:); %bierzemy pierwsze 70% macierzy na zbior uczacy
% training_labels = table2array(training_T(:,2)); % labele zbioru uczacego - znajduja sie w 2 kolumnie
% training_T = table2array(training_T(:,3:end)); % parametry uczace - od 3 kolumny do konca (w pierwszej jest l. porzadkowa, wiec nas nie interesuje, a w 2 sa klasy
% validation_T = random_T(floor(0.7*size(random_T,1))+1:floor(0.85*size(random_T,1)),:); %zbior walidacyjny - 15%
% validation_labels = table2array(validation_T(:,2));
% validation_T = table2array(validation_T(:,3:end));
% testing_T = random_T(floor(0.85*size(random_T,1))+1:end,:); %zbior testowy - 15%
% testing_labels = table2array(testing_T(:,2));
% testing_T = table2array(testing_T(:,3:end));
% mean(training_labels) %sprawdzamy, czy mamy taki sam udzial obu klas w kazdym z trzech podzbiorow
% mean(validation_labels)
% mean(testing_labels)
% 
% mdl = fitglm(training_T,training_labels,'Distribution','binomial','Link','logit');
% %zbior walidacyjnuy
% scores=mdl.predict(validation_T);
% [X,Y,TH,AUC] = perfcurve(validation_labels,scores,'1');
% AUC
% plot(X,Y)
% xlabel('False positive rate') 
% ylabel('True positive rate')
% title('ROC for Classification by Logistic Regression - validation set')
% %zbior testowy
% scores=mdl.predict(testing_T);
% [X,Y,TH,AUC] = perfcurve(testing_labels,scores,'1');
% AUC
% plot(X,Y)
% xlabel('False positive rate') 
% ylabel('True positive rate')
% title('ROC for Classification by Logistic Regression - testing set')
% 
% scores(scores>=0.5) = 1;
% scores(scores<0.5) = 0;
% 
% [C,order] = confusionmat(testing_labels,scores)  %macierzpomylek
% cm_test = confusionchart(testing_labels,scores) %ladniejsza wersja macierzy - taka kolorowa jak byla pokazana wyzej
% TP(i) = C(1,1)
% FP(i) = C(2,1)
% TN(i) = C(2,2)
% FN(i) = C(1,2)
% 
% %tu proszewyliczyc 5 popularnych metryk, ktore wymienione sapowyzej (pod
% %macierzapomyleksa wzory)
% TPR(i)=TP(i)/(TP(i)+FN(i)) %recall (czulosc)
% TNR(i)=TN(i)/(TN(i)+FP(i)) %specificity (swoistosc)
% PPV(i)=TP(i)/(TP(i)+FP(i)) %precision (precyzja)
% ACC(i)=(TP(i)+TN(i))/(TP(i)+TN(i)+FP(i)+FN(i)) %accuracy (dokladnosc)
% F1(i)=2*((PPV(i)*TPR(i))/(PPV(i)+TPR(i))) %f1 score
% end
% 
% k=round(sqrt(120/2)) %ustalenie liczba klas dla warunku powyzej 100 prob
% figure
% subplot(2,3,1)
% h1=histogram(TPR,k)
% xlabel('czulosc')
% ylabel('liczebnosc klas')
% subplot(2,3,2)
% h2=histogram(TNR,k)
% xlabel('swoistosc')
% ylabel('liczebnosc klas')
% subplot(2,3,3)
% h3=histogram(PPV,k)
% xlabel('precyzja')
% ylabel('liczebnosc klas')
% subplot(2,3,4)
% h4=histogram(ACC,k)
% xlabel('dokladnosc')
% ylabel('liczebnosc klas')
% subplot(2,3,5)
% h5=histogram(F1,k)
% xlabel('f1-score')
% ylabel('liczebnosc klas')
% 
% figure 
% subplot(2,3,1)
% p1=probplot(TPR)
% xlabel('czulosc')
% sr1=mean(TPR)
% odch1=std(TPR)
% k1=kurtosis(TPR)
% skew1=skewness(TPR)
% [h,p,kstat,critval] = lillietest(TPR,'alpha',0.05)
% subplot(2,3,2)
% p2=probplot(TNR)
% xlabel('swoistosc')
% sr2=mean(TNR)
% odch2=std(TNR)
% k2=kurtosis(TNR)
% skew2=skewness(TNR)
% [h,p,kstat,critval] = lillietest(TNR,'alpha',0.05)
% subplot(2,3,3)
% p3=probplot(PPV)
% xlabel('precyzja')
% sr3=mean(PPV)
% odch3=std(PPV)
% k3=kurtosis(PPV)
% skew3=skewness(PPV)
% [h,p,kstat,critval] = lillietest(PPV,'alpha',0.05)
% subplot(2,3,4)
% p4=probplot(ACC)
% xlabel('dokladnosc')
% sr4=mean(ACC)
% odch4=std(ACC)
% k4=kurtosis(ACC)
% skew4=skewness(ACC)
% [h,p,kstat,critval] = lillietest(ACC,'alpha',0.05)
% subplot(2,3,5)
% p5=probplot(F1)
% xlabel('f1-score')
% sr5=mean(F1)
% odch5=std(F1)
% k5=kurtosis(F1)
% skew5=skewness(F1)
% [h,p,kstat,critval] = lillietest(F1,'alpha',0.05)

%Proba 2

% clc
% clear all;
% T = readtable('titanic.csv');
% A2 = T.Variables
% 
% A1=nanmean(A2(:,5))
% A2(isnan(A2))=A1
% T=array2table(A2)
% for i=1:120
% 
% random_T = T(randperm(size(T, 1)), :); %zmieniamy kolejnoscobiektow na losowa, zebyuniknacduzychskupien jednej klasy
% training_T = random_T(1:floor(0.7*size(random_T,1)),:); %bierzemy pierwsze 70% macierzy na zbioruczacy
% training_labels = table2array(training_T(:,2)); % labele zbioru uczacego - znajdujasie w 2 kolumnie
% training_T = table2array(training_T(:,3:end)); % parametry uczace - od 3 kolumny do konca (w pierwszej jest l. porzadkowa, wiec nas nie interesuje, a w 2 sa klasy
% validation_T = random_T(floor(0.7*size(random_T,1))+1:floor(0.85*size(random_T,1)),:); %zbiorwalidacyjny - 15%
% validation_labels = table2array(validation_T(:,2));
% validation_T = table2array(validation_T(:,3:end));
% testing_T = random_T(floor(0.85*size(random_T,1))+1:end,:); %zbiortestowy - 15%
% testing_labels = table2array(testing_T(:,2));
% testing_T = table2array(testing_T(:,3:end));
% mean(training_labels) %sprawdzamy, czy mamy taki sam udzial obu klas w kazdym z trzech podzbiorow
% mean(validation_labels)
% mean(testing_labels)
% 
% mdl = fitglm(training_T,training_labels,'Distribution','binomial','Link','logit');
% %zbiorwalidacyjnuy
% scores=mdl.predict(validation_T);
% [X,Y,TH,AUC] = perfcurve(validation_labels,scores,'1');
% AUC
% plot(X,Y)
% xlabel('False positive rate') 
% ylabel('True positive rate')
% title('ROC for Classification by Logistic Regression - validation set')
% %zbiortestowy
% scores=mdl.predict(testing_T);
% [X,Y,TH,AUC] = perfcurve(testing_labels,scores,'1');
% AUC
% plot(X,Y)
% xlabel('False positive rate') 
% ylabel('True positive rate')
% title('ROC for Classification by Logistic Regression - testing set')
% 
% scores(scores>=0.5) = 1;
% scores(scores<0.5) = 0;
% [C,order] = confusionmat(testing_labels,scores)  %macierzpomylek
% %cm_test = confusionchart(testing_labels,scores) %ladniejsza wersja macierzy - taka kolorowa jak byla pokazana wyzej
% TP(i) = C(1,1)
% FP(i) = C(2,1)
% TN(i) = C(2,2)
% FN(i) = C(1,2)
% 
% %tu proszewyliczyc 5 popularnych metryk, ktore wymienione sapowyzej (pod
% %macierzapomyleksa wzory)
% TPR(i)=TP(i)/(TP(i)+FN(i)) %recall (czulosc)
% TNR(i)=TN(i)/(TN(i)+FP(i)) %specificity (swoistosc)
% PPV(i)=TP(i)/(TP(i)+FP(i)) %precision (precyzja)
% ACC(i)=(TP(i)+TN(i))/(TP(i)+TN(i)+FP(i)+FN(i)) %accuracy (dokladnosc)
% F1(i)=2*((PPV(i)*TPR(i))/(PPV(i)+TPR(i))) %f1 score
% end
% 
% k=round(sqrt(120/2)) %ustalenie liczba klas dla warunku powyzej 100 prob
% figure
% subplot(2,3,1)
% h1=histogram(TPR,k)
% xlabel('czulosc')
% ylabel('liczebnosc klas')
% subplot(2,3,2)
% h2=histogram(TNR,k)
% xlabel('swoistosc')
% ylabel('liczebnosc klas')
% subplot(2,3,3)
% h3=histogram(PPV,k)
% xlabel('precyzja')
% ylabel('liczebnosc klas')
% subplot(2,3,4)
% h4=histogram(ACC,k)
% xlabel('dokladnosc')
% ylabel('liczebnosc klas')
% subplot(2,3,5)
% h5=histogram(F1,k)
% xlabel('f1-score')
% ylabel('liczebnosc klas')
% 
% figure 
% subplot(2,3,1)
% p1=probplot(TPR)
% xlabel('czulosc')
% sr1=mean(TPR)
% odch1=std(TPR)
% k1=kurtosis(TPR)
% skew1=skewness(TPR)
% [h,p,kstat,critval] = lillietest(TPR,'alpha',0.05)
% subplot(2,3,2)
% p2=probplot(TNR)
% xlabel('swoistosc')
% sr2=mean(TNR)
% odch2=std(TNR)
% k2=kurtosis(TNR)
% skew2=skewness(TNR)
% [h,p,kstat,critval] = lillietest(TNR,'alpha',0.05)
% subplot(2,3,3)
% p3=probplot(PPV)
% xlabel('precyzja')
% sr3=mean(PPV)
% odch3=std(PPV)
% k3=kurtosis(PPV)
% skew3=skewness(PPV)
% [h,p,kstat,critval] = lillietest(PPV,'alpha',0.05)
% subplot(2,3,4)
% p4=probplot(ACC)
% xlabel('dokladnosc')
% sr4=mean(ACC)
% odch4=std(ACC)
% k4=kurtosis(ACC)
% skew4=skewness(ACC)
% [h,p,kstat,critval] = lillietest(ACC,'alpha',0.05)
% subplot(2,3,5)
% p5=probplot(F1)
% xlabel('f1-score')
% sr5=mean(F1)
% odch5=std(F1)
% k5=kurtosis(F1)
% skew5=skewness(F1)
% [h,p,kstat,critval] = lillietest(F1,'alpha',0.05)

% %Proba 3
% 
% clc
% clear all;
% T = readtable('titanic.csv');
% %bez kolumny wieku
% T=T(:,[1:4,6:end])
% 
% for i=1:120
% random_T = T(randperm(size(T, 1)), :); %zmieniamy kolejnosc obiektow na losowa, zeby uniknac duzych skupien jednej klasy
% training_T = random_T(1:floor(0.35*size(random_T,1)),:); %bierzemy pierwsze 70% macierzy na zbior uczacy
% training_labels = table2array(training_T(:,2)); % labele zbioru uczacego - znajduja sie w 2 kolumnie
% training_T = table2array(training_T(:,3:end)); % parametry uczace - od 3 kolumny do konca (w pierwszej jest l. porzadkowa, wiec nas nie interesuje, a w 2 sa klasy
% validation_T = random_T(floor(0.35*size(random_T,1))+1:floor(0.425*size(random_T,1)),:); %zbior walidacyjny - 15%
% validation_labels = table2array(validation_T(:,2));
% validation_T = table2array(validation_T(:,3:end));
% testing_T = random_T(floor(0.425*size(random_T,1))+1:end,:); %zbior testowy - 15%
% testing_labels = table2array(testing_T(:,2));
% testing_T = table2array(testing_T(:,3:end));
% mean(training_labels) %sprawdzamy, czy mamy taki sam udzial obu klas w kazdym z trzech podzbiorow
% mean(validation_labels)
% mean(testing_labels)
% 
% mdl = fitglm(training_T,training_labels,'Distribution','binomial','Link','logit');
% %zbior walidacyjnuy
% scores=mdl.predict(validation_T);
% [X,Y,TH,AUC] = perfcurve(validation_labels,scores,'1');
% AUC
% plot(X,Y)
% xlabel('False positive rate') 
% ylabel('True positive rate')
% title('ROC for Classification by Logistic Regression - validation set')
% %zbior testowy
% scores=mdl.predict(testing_T);
% [X,Y,TH,AUC] = perfcurve(testing_labels,scores,'1');
% AUC
% plot(X,Y)
% xlabel('False positive rate') 
% ylabel('True positive rate')
% title('ROC for Classification by Logistic Regression - testing set')
% 
% scores(scores>=0.5) = 1; 
% scores(scores<0.5) = 0;
% [C,order] = confusionmat(testing_labels,scores)  %macierzpomylek
% %cm_test = confusionchart(testing_labels,scores) %ladniejsza wersja macierzy - taka kolorowa jak byla pokazana wyzej
% TP(i) = C(1,1)
% FP(i) = C(2,1)
% TN(i) = C(2,2)
% FN(i) = C(1,2)
% 
% %tu proszewyliczyc 5 popularnych metryk, ktore wymienione sapowyzej (pod
% %macierzapomyleksa wzory)
% TPR(i)=TP(i)/(TP(i)+FN(i)) %recall (czulosc)
% TNR(i)=TN(i)/(TN(i)+FP(i)) %specificity (swoistosc)
% PPV(i)=TP(i)/(TP(i)+FP(i)) %precision (precyzja)
% ACC(i)=(TP(i)+TN(i))/(TP(i)+TN(i)+FP(i)+FN(i)) %accuracy (dokladnosc)
% F1(i)=2*((PPV(i)*TPR(i))/(PPV(i)+TPR(i))) %f1 score
% end
% 
% k=round(sqrt(120/2)) %ustalenie liczba klas dla warunku powyzej 100 prob
% figure
% subplot(2,3,1)
% h1=histogram(TPR,k)
% xlabel('czulosc')
% ylabel('liczebnosc klas')
% subplot(2,3,2)
% h2=histogram(TNR,k)
% xlabel('swoistosc')
% ylabel('liczebnosc klas')
% subplot(2,3,3)
% h3=histogram(PPV,k)
% xlabel('precyzja')
% ylabel('liczebnosc klas')
% subplot(2,3,4)
% h4=histogram(ACC,k)
% xlabel('dokladnosc')
% ylabel('liczebnosc klas')
% subplot(2,3,5)
% h5=histogram(F1,k)
% xlabel('f1-score')
% ylabel('liczebnosc klas')
% 
% figure 
% subplot(2,3,1)
% p1=probplot(TPR)
% xlabel('czulosc')
% sr1=mean(TPR)
% odch1=std(TPR)
% k1=kurtosis(TPR)
% skew1=skewness(TPR)
% [h,p,kstat,critval] = lillietest(TPR,'alpha',0.05)
% subplot(2,3,2)
% p2=probplot(TNR)
% xlabel('swoistosc')
% sr2=mean(TNR)
% odch2=std(TNR)
% k2=kurtosis(TNR)
% skew2=skewness(TNR)
% [h,p,kstat,critval] = lillietest(TNR,'alpha',0.05)
% subplot(2,3,3)
% p3=probplot(PPV)
% xlabel('precyzja')
% sr3=mean(PPV)
% odch3=std(PPV)
% k3=kurtosis(PPV)
% skew3=skewness(PPV)
% [h,p,kstat,critval] = lillietest(PPV,'alpha',0.05)
% subplot(2,3,4)
% p4=probplot(ACC)
% xlabel('dokladnosc')
% sr4=mean(ACC)
% odch4=std(ACC)
% k4=kurtosis(ACC)
% skew4=skewness(ACC)
% [h,p,kstat,critval] = lillietest(ACC,'alpha',0.05)
% subplot(2,3,5)
% p5=probplot(F1)
% xlabel('f1-score')
% sr5=mean(F1)
% odch5=std(F1)
% k5=kurtosis(F1)
% skew5=skewness(F1)
% [h,p,kstat,critval] = lillietest(F1,'alpha',0.05)
% 
% %próba 4
% 
% clc 
% clear all;
% 
% T = readtable('titanic.csv');
% A2 = T.Variables
% 
% A1=nanmean(A2(:,5))
% A2(isnan(A2))=A1
% T=array2table(A2)
% 
% for i=1:120
% random_T = T(randperm(size(T, 1)), :); %zmieniamy kolejnosc obiektow na losowa, zeby uniknac duzych skupien jednej klasy
% training_T = random_T(1:floor(0.35*size(random_T,1)),:); %bierzemy pierwsze 70% macierzy na zbior uczacy
% training_labels = table2array(training_T(:,2)); % labele zbioru uczacego - znajduja sie w 2 kolumnie
% training_T = table2array(training_T(:,3:end)); % parametry uczace - od 3 kolumny do konca (w pierwszej jest l. porzadkowa, wiec nas nie interesuje, a w 2 sa klasy
% validation_T = random_T(floor(0.35*size(random_T,1))+1:floor(0.425*size(random_T,1)),:); %zbior walidacyjny - 15%
% validation_labels = table2array(validation_T(:,2));
% validation_T = table2array(validation_T(:,3:end));
% testing_T = random_T(floor(0.425*size(random_T,1))+1:end,:); %zbior testowy - 15%
% testing_labels = table2array(testing_T(:,2));
% testing_T = table2array(testing_T(:,3:end));
% mean(training_labels) %sprawdzamy, czy mamy taki sam udzial obu klas w kazdym z trzech podzbiorow
% mean(validation_labels)
% mean(testing_labels)
% 
% mdl = fitglm(training_T,training_labels,'Distribution','binomial','Link','logit');
% %zbior walidacyjnuy
% scores=mdl.predict(validation_T);
% [X,Y,TH,AUC] = perfcurve(validation_labels,scores,'1');
% AUC
% plot(X,Y)
% xlabel('False positive rate') 
% ylabel('True positive rate')
% title('ROC for Classification by Logistic Regression - validation set')
% %zbior testowy
% scores=mdl.predict(testing_T);
% [X,Y,TH,AUC] = perfcurve(testing_labels,scores,'1');
% AUC
% plot(X,Y)
% xlabel('False positive rate') 
% ylabel('True positive rate')
% title('ROC for Classification by Logistic Regression - testing set')
% 
% scores(scores>=0.5) = 1; 
% scores(scores<0.5) = 0;
% [C,order] = confusionmat(testing_labels,scores)  %macierz pomylek
% %cm_test = confusionchart(testing_labels,scores) %ladniejsza wersja macierzy - taka kolorowa jak byla pokazana wyzej
% TP(i) = C(1,1)
% FP(i) = C(2,1)
% TN(i) = C(2,2)
% FN(i) = C(1,2)
% 
% %tu proszewyliczyc 5 popularnych metryk, ktore wymienione sapowyzej (pod
% %macierzapomyleksa wzory)
% TPR(i)=TP(i)/(TP(i)+FN(i)) %recall (czulosc)
% TNR(i)=TN(i)/(TN(i)+FP(i)) %specificity (swoistosc)
% PPV(i)=TP(i)/(TP(i)+FP(i)) %precision (precyzja)
% ACC(i)=(TP(i)+TN(i))/(TP(i)+TN(i)+FP(i)+FN(i)) %accuracy (dokladnosc)
% F1(i)=2*((PPV(i)*TPR(i))/(PPV(i)+TPR(i))) %f1 score
% end
% 
% k=round(sqrt(120/2)) %ustalenie liczba klas dla warunku powyzej 100 prob
% figure
% subplot(2,3,1)
% h1=histogram(TPR,k)
% xlabel('czulosc')
% ylabel('liczebnosc klas')
% subplot(2,3,2)
% h2=histogram(TNR,k)
% xlabel('swoistosc')
% ylabel('liczebnosc klas')
% subplot(2,3,3)
% h3=histogram(PPV,k)
% xlabel('precyzja')
% ylabel('liczebnosc klas')
% subplot(2,3,4)
% h4=histogram(ACC,k)
% xlabel('dokladnosc')
% ylabel('liczebnosc klas')
% subplot(2,3,5)
% h5=histogram(F1,k)
% xlabel('f1-score')
% ylabel('liczebnosc klas')
% 
% figure 
% subplot(2,3,1)
% p1=probplot(TPR)
% xlabel('czulosc')
% sr1=mean(TPR)
% odch1=std(TPR)
% k1=kurtosis(TPR)
% skew1=skewness(TPR)
% [h,p,kstat,critval] = lillietest(TPR,'alpha',0.05)
% subplot(2,3,2)
% p2=probplot(TNR)
% xlabel('swoistosc')
% sr2=mean(TNR)
% odch2=std(TNR)
% k2=kurtosis(TNR)
% skew2=skewness(TNR)
% [h,p,kstat,critval] = lillietest(TNR,'alpha',0.05)
% subplot(2,3,3)
% p3=probplot(PPV)
% xlabel('precyzja')
% sr3=mean(PPV)
% odch3=std(PPV)
% k3=kurtosis(PPV)
% skew3=skewness(PPV)
% [h,p,kstat,critval] = lillietest(PPV,'alpha',0.05)
% subplot(2,3,4)
% p4=probplot(ACC)
% xlabel('dokladnosc')
% sr4=mean(ACC)
% odch4=std(ACC)
% k4=kurtosis(ACC)
% skew4=skewness(ACC)
% [h,p,kstat,critval] = lillietest(ACC,'alpha',0.05)
% subplot(2,3,5)
% p5=probplot(F1)
% xlabel('f1-score')
% sr5=mean(F1)
% odch5=std(F1)
% k5=kurtosis(F1)
% skew5=skewness(F1)
% [h,p,kstat,critval] = lillietest(F1,'alpha',0.05)
%


%analiza wynikow porownuje z soba te same parametry ale z roznych prob

clc
clear all;
T = readtable('wynik1.csv');
s=table2array(T)
%bez kolumny wieku
% [h,p,kstat,critval] = lillietest(s(:,1),'alpha',0.05)
% [h,p,kstat,critval] = lillietest(s(:,2),'alpha',0.05)
% [h,p,kstat,critval] = lillietest(s(:,3),'alpha',0.05)
% [h,p,kstat,critval] = lillietest(s(:,4),'alpha',0.05)
% [h,p,kstat,critval] = lillietest(s(:,5),'alpha',0.05)
% [h,p,kstat,critval] = lillietest(s(:,6),'alpha',0.05)
% [h,p,kstat,critval] = lillietest(s(:,7),'alpha',0.05)
% [h,p,kstat,critval] = lillietest(s(:,8),'alpha',0.05)
% [h,p,kstat,critval] = lillietest(s(:,9),'alpha',0.05)
% [h,p,kstat,critval] = lillietest(s(:,10),'alpha',0.05)
% [h,p,kstat,critval] = lillietest(s(:,11),'alpha',0.05)
% [h,p,kstat,critval] = lillietest(s(:,12),'alpha',0.05)
% [h,p,kstat,critval] = lillietest(s(:,13),'alpha',0.05)
% [h,p,kstat,critval] = lillietest(s(:,14),'alpha',0.05)
% [h,p,kstat,critval] = lillietest(s(:,15),'alpha',0.05)
% [h,p,kstat,critval] = lillietest(s(:,16),'alpha',0.05)
% [h,p,kstat,critval] = lillietest(s(:,17),'alpha',0.05)
% [h,p,kstat,critval] = lillietest(s(:,18),'alpha',0.05)
% [h,p,kstat,critval] = lillietest(s(:,19),'alpha',0.05)
% [h,p,kstat,critval] = lillietest(s(:,20),'alpha',0.05)

% W2=T(:,[2,7,12,17])

% subplot(2,2,1)
% h1=histogram(s(:,2),8)
% grid on
% title('metoda 1')
% xlabel('czulosc')
% ylabel('liczebnosc klas')
% axis([0.5 0.9 0 40])
% subplot(2,2,2)
% h2=histogram(s(:,7),8)
% grid on
% title('metoda 2')
% xlabel('czulosc')
% ylabel('liczebnosc klas')
% axis([0.5 0.9 0 40])
% subplot(2,2,3)
% h3=histogram(s(:,12),8)
% grid on
% title('metoda 3')
% xlabel('czulosc')
% ylabel('liczebnosc klas')
% axis([0.5 0.9 0 40])
% subplot(2,2,4)
% h4=histogram(s(:,17),8)
% grid on
% title('metoda 4')
% xlabel('czulosc')
% ylabel('liczebnosc klas')
% axis([0.5 0.9 0 40])

[p,tbl,stats] = anova1(s(:,[2,7,12,17]));
xlabel('metody')
figure
xlim([ 0.5 5.5])
ylim([0.5 33.0])
multcompare(stats) 
ylabel('metody')






