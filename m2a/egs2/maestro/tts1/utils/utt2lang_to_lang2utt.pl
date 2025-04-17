#!/usr/bin/env perl
# Copyright 2010-2011 Microsoft Corporation

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
# WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
# MERCHANTABLITY OR NON-INFRINGEMENT.
# See the Apache 2 License for the specific language governing permissions and
# limitations under the License.

# converts an utt2lang file to a lang2utt file.
# Takes input from the stdin or from a file argument;
# output goes to the standard out.

if ( @ARGV > 1 ) {
    die "Usage: utt2lang_to_lang2utt.pl [ utt2lang ] > lang2utt";
}

while(<>){ 
    @A = split(" ", $_);
    @A == 2 || die "Invalid line in utt2lang file: $_";
    ($u,$s) = @A;
    if(!$seen_lang{$s}) {
        $seen_lang{$s} = 1;
        push @langlist, $s;
    }
    push (@{$lang_hash{$s}}, "$u");
}
foreach $s (@langlist) {
    $l = join(' ',@{$lang_hash{$s}});
    print "$s $l\n";
}
